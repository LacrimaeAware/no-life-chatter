# services/message_service.py
import logging
import aiohttp
import asyncio
from google.oauth2 import service_account
from google.cloud import translate_v2 as translate
from utils.user_settings import get_user_settings
from utils.romanize import romanize, is_supported, thai_segment
from utils.language_detect import detect_language
import re
import sqlite3
import html

import config

# ----------------- helpers -----------------

def fetch_usernames(channel_name):
    db_path = config.DB_PATH
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT user_name FROM channel_users WHERE channel_name = ?", (channel_name,))
            return [row[0].lower() for row in cur.fetchall()]
    except Exception as e:
        logging.error(f"Failed to fetch usernames: {e}")
        return []

def filter_usernames_from_message(message, usernames):
    if not usernames:
        return message
    pattern = r'\b(' + '|'.join(re.escape(n) for n in usernames) + r')\b'
    return re.sub(pattern, '', message, flags=re.IGNORECASE).strip()

def log_user_activity(channel_name, user_name):
    db_path = config.DB_PATH
    user_name = user_name.lower()
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO channel_users (channel_name, user_name, last_active)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(channel_name, user_name) DO UPDATE SET last_active = CURRENT_TIMESTAMP;
        """, (channel_name, user_name))
        conn.commit()

def decode_html_entities(text):
    return html.unescape(text)

def remove_at_words(text):
    return re.sub(r"@\S+", "", text).lstrip()

def is_translation_enabled_globally():
    db_path = config.DB_PATH
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT setting_value FROM global_settings WHERE setting_name = 'global_translation'")
        row = cur.fetchone()
        return row[0] if row else False

def is_translation_enabled_for_channel(channel_name):
    db_path = config.DB_PATH
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT translation_enabled FROM channel_settings WHERE channel_name = ?", (channel_name,))
        row = cur.fetchone()
        return row[0] if row else False

# ----------------- service -----------------

class MessageService:
    def __init__(self, bot):
        self.bot = bot
        credentials = service_account.Credentials.from_service_account_file(
            config.GOOGLE_CREDENTIALS
        )
        self.translator = translate.Client(credentials=credentials)

    # lang utilities
    def detect_lang(self, text):
        # Local, free detection (no API cost). Translation still uses Google.
        return detect_language(text)

    def gtranslate(self, text, target):
        out = self.translator.translate(text, target_language=target.upper())
        return decode_html_entities(out.get('translatedText', '') or '')

    async def handle_regular_message(self, message):
        logging.info(f"Handling regular message from {message.author.name}: {message.content}")

        user_settings = get_user_settings(message.author.id)
        log_user_activity(message.channel.name, message.author.name)

        # cleaning
        usernames = fetch_usernames(message.channel.name)
        msg = filter_usernames_from_message(remove_at_words(message.content), usernames)

        # nothing meaningful left after cleaning
        if not msg:
            return

        # starts with non-alpha? skip (emotes, punctuation-only, etc.)
        if not msg[0].isalpha():
            return

        # scrub URLs
        msg = re.sub(r'\b(?:https?://)?\S+\.\S+\b', '***', msg)

        # ===== PRACTICE MODE (CSV in learn_lang, separate whispers) =====
        if user_settings and user_settings.get('practice_mode'):
            native = (user_settings.get('native_lang') or 'EN').upper()
            learn_csv = (user_settings.get('learn_lang') or 'ES')
            learn_list = [x.strip().upper() for x in learn_csv.split(',') if x.strip()]

            # never practice the native language by mistake
            learn_list = [L for L in learn_list if L != native]
            if not learn_list:
                # nothing to learn -> do nothing
                return

            lang, conf = self.detect_lang(msg)
            if conf < config.MIN_CONFIDENCE:
                return

            # If user wrote in a learn language -> post native to channel (for others)
            if lang in learn_list:
                translated = self.gtranslate(msg, native)
                if translated and translated != msg:
                    await message.channel.send(translated)
                return

            # If user wrote in native -> whisper EACH learn translation separately
            if lang == native:
                romanize_on = bool(user_settings.get('romanize_enabled', 0))
                from utils.token_manager import get_current_helix_token
                from utils.romanize import is_supported, romanize, thai_segment, supported_languages
                token = get_current_helix_token()
                WHISPER_DELAY = 0.75  # seconds between whispers

                # Normalize common aliases users may type
                ALIASES = {"KR": "KO", "JP": "JA"}
                learn_list = [ALIASES.get(L, L) for L in learn_list]

                # Translate once per language
                translations = {}
                for L in learn_list:
                    try:
                        t = self.gtranslate(msg, L)
                    except Exception as e:
                        logging.warning(f"Translation to {L} failed: {e}")
                        t = None
                    if t and t != msg:
                        translations[L] = t

                # For each language:
                for L in learn_list:
                    t = translations.get(L)
                    if not t:
                        continue

                    # Thai translation gets segmentation for readability
                    if L == 'TH':
                        seg = thai_segment(t)
                        if seg and seg != t:
                            t = seg

                    # 1) Translation whisper (always)
                    await self.send_whisper(token, message.author.id, f"{L}: {t}")
                    await asyncio.sleep(WHISPER_DELAY)

                    # 2) Romanization whisper (only if the language has a romanizer)
                    if romanize_on and is_supported(L):
                        try:
                            rom = romanize(L, t)
                        except Exception as e:
                            logging.warning(f"Romanization for {L} failed: {e}")
                            rom = None
                        if rom and rom != t:
                            await self.send_whisper(token, message.author.id, f"{L}-ROM: {rom}")
                            await asyncio.sleep(WHISPER_DELAY)

                return

            # otherwise, ignore in practice mode
            return

        # ===== original flows =====
        if user_settings:
            if user_settings.get('translation_enabled'):
                target = user_settings.get('translation_language', 'EN')
                t = self.gtranslate(msg, target)
                mode = user_settings.get('output_mode', 'default')
                await self.output_message(message, t or "Failed to translate message.", mode, user_settings)
            else:
                if ' ' in msg and is_translation_enabled_globally() and is_translation_enabled_for_channel(message.channel.name):
                    await self.handle_translation_if_needed(message, 'EN')
        else:
            if ' ' in msg and is_translation_enabled_globally() and is_translation_enabled_for_channel(message.channel.name):
                await self.handle_translation_if_needed(message, 'EN')

    async def output_message(self, message, content, mode, user_settings):
        if mode == 'whisper':
            from utils.token_manager import get_current_helix_token
            await self.send_whisper(get_current_helix_token(), message.author.id, content)
        elif mode == 'channel':
            channel_name = (user_settings.get('output_channel') if user_settings else None)
            channel = None
            if channel_name:
                channel_name = channel_name.lower().lstrip('#').lstrip('~')
                channel = self.bot.get_channel(channel_name)
            if not channel:
                channel = message.channel
            await channel.send(content)
        else:
            await message.channel.send(content)

    async def handle_translation_if_needed(self, message, target_language):
        lang, conf = self.detect_lang(message.content)
        if conf < config.MIN_CONFIDENCE or lang not in config.SUPPORTED_LANGS:
            return
        if lang != target_language.upper():
            t = self.translator.translate(message.content, target_language=target_language.upper())
            txt = decode_html_entities(t.get('translatedText', '') or '')
            if txt and txt != message.content:
                await message.channel.send(txt)

    async def send_whisper(self, token, user_id, message):
        url = "https://api.twitch.tv/helix/whispers"
        headers = {
            'Authorization': f'Bearer {token}',
            'Client-ID': self.bot.client_id,
            'Content-Type': 'application/json'
        }
        payload = {
            'from_user_id': str(self.bot.user_id),
            'to_user_id': str(user_id),
            'message': message
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if 200 <= resp.status < 300:
                    logging.info(f"Whisper sent OK ({resp.status}) to user {user_id}")
                else:
                    txt = await resp.text()
                    logging.error(f"Failed to send whisper ({resp.status}): {txt}")