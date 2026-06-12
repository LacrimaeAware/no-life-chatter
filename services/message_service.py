# services/message_service.py
import logging
import aiohttp
import asyncio
import random
import time
from google.oauth2 import service_account
from google.cloud import translate_v2 as translate
from utils.user_settings import get_user_settings
from utils.romanize import romanize, is_supported, thai_segment
from utils.language_detect import detect_language, detect_confidences
from services.translators import deepl_translate
from services.emotes import ensure_channel_emotes, strip_emotes
from utils.speaker_profile import known_languages, record_language
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
    # Only strip bare usernames that are long enough to be unambiguous — short
    # ones (<= 3 chars) collide with real words and would mangle messages.
    # (@mentions are already handled separately by remove_at_words.)
    names = [n for n in usernames if len(n) >= 4]
    if not names:
        return message
    pattern = r'\b(' + '|'.join(re.escape(n) for n in names) + r')\b'
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

def strip_emote_like_tokens(text):
    """Drop tokens that look like emote names rather than words.

    Emotes not in the fetched 7TV/BTTV/FFZ lists (and native Twitch emotes,
    which the bot doesn't enumerate) reach detection as plain text, and both
    lingua and DeepL treat them as foreign words — e.g. "CuldBeWorthIt" detects
    as German 0.63 and DeepL turns it into "Is It Worth It?". Such names are
    structurally distinct from real words: an internal capital (CamelCase, like
    "CuldBeWorthIt" / "hesRight") or a long all-caps run with no spaces (like
    "TELLMEHEDIDNTJUSTSAYTHAT"). Ordinary chat words — including Title Case and
    short shouting — don't look like that, so dropping these before detection is
    safe and keeps any real words around them.
    """
    kept = []
    for tok in text.split():
        core = tok.strip('.,!?;:()[]')
        if re.search(r'[a-z][A-Z]', core):           # CamelCase emote name
            continue
        letters = [c for c in core if c.isalpha()]
        if len(letters) >= 12 and core.isupper():    # long all-caps mash-emote
            continue
        kept.append(tok)
    return ' '.join(kept)

def is_stop_followup(text):
    return (text or "").strip().upper() in {"STOP", "NO", "NONE"}

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
        # Translation backends are optional. DeepL (a key in .env) is tried
        # first; Google (a service-account file) is an optional fallback. The
        # bot still runs with neither — translation features just stay quiet.
        self.deepl_enabled = bool(config.DEEPL_API_KEY)
        self.translator = None
        try:
            credentials = service_account.Credentials.from_service_account_file(
                config.GOOGLE_CREDENTIALS
            )
            self.translator = translate.Client(credentials=credentials)
        except Exception as e:
            logging.info(f"Google translation not configured ({e}).")

        self._last_reaction = {}  # channel -> last persona-reaction timestamp
        self.can_translate = self.deepl_enabled or (self.translator is not None)
        if self.deepl_enabled:
            logging.info("Translation: DeepL enabled.")
        elif self.translator is not None:
            logging.info("Translation: Google enabled.")
        else:
            logging.warning("Translation disabled — no DeepL key or Google credentials.")

    # lang utilities
    def detect_lang(self, text):
        # Local, free detection (no API cost). Translation still uses Google.
        return detect_language(text)

    def gtranslate(self, text, target):
        # DeepL first (free tier), Google as fallback.
        out = deepl_translate(text, target)
        if out:
            return decode_html_entities(out)
        if self.translator is not None:
            g = self.translator.translate(text, target_language=target.upper())
            return decode_html_entities(g.get('translatedText', '') or '')
        return None

    def _persona_name_variants(self, target):
        variants = {target.lower()}
        if "_" in target:
            short = target.split("_", 1)[0].lower()
            if len(short) >= 4:
                variants.add(short)
        return variants

    def _directed_persona_targets(self, content, authors):
        text = (content or "").lower()
        directed = []
        for target in authors:
            for variant in self._persona_name_variants(target):
                if f"@{variant}" in text:
                    directed.append(target)
                    break
                if re.search(rf"(?<![a-z0-9_]){re.escape(variant)}(?![a-z0-9_])", text):
                    directed.append(target)
                    break
        return directed

    async def _persona_line(self, target, channel, is_clean, persona_llm):
        cand = await persona_llm.generate(
            target, channel, mode="normal", copy_strategy="drop",
            invoked_by="ambient-reaction",
        )
        if cand and len(cand.split()) >= 2 and is_clean(cand):
            return cand
        return None

    async def _maybe_continue_reaction(self, message, target, first_line, is_clean, persona_llm):
        chance = getattr(config, "REACTION_CONTINUE_CHANCE", 0.0)
        max_lines = max(0, getattr(config, "REACTION_MAX_CONTINUATIONS", 0))
        if chance <= 0 or max_lines <= 0:
            return

        last_line = first_line
        for _ in range(max_lines):
            if random.random() >= chance:
                break
            await asyncio.sleep(getattr(config, "REACTION_CONTINUE_DELAY", 1.5))
            follow_prompt = (
                f'You just said: "{last_line}". Decide if {target} would '
                f"naturally send one immediate second chat message that "
                f"continues the SAME thought. If yes, output only that "
                f"short follow-up. If no coherent follow-up is natural, "
                f"output exactly STOP. Do not change topic."
            )
            line = await persona_llm.generate(
                target,
                message.channel.name,
                follow_prompt,
                mode="normal",
                exemplar_count=getattr(config, "LLM_RETRY_EXEMPLARS", 60),
                context_count=getattr(config, "LLM_RETRY_CONTEXT", 12),
            )
            if not line or is_stop_followup(line) or not is_clean(line):
                break
            if len(line) > 280:
                line = line[:279] + "..."
            tag = persona_llm.last_model_tag()
            prefix = f"#{tag} " if tag else ""
            logging.info(f"Persona reaction follow-up in #{message.channel.name} as {target}: {line!r}")
            await message.channel.send(f"↳ 🎲 {prefix}{target}: {line}")
            last_line = line

    async def maybe_react(self, message):
        """Rarely, post an LLM persona line of a random recent chatter.

        This is the "the bot randomly does a bit" feature. Markov is kept
        explicit-command only via ~mimic/~markov.
        """
        channel = message.channel.name
        try:
            from utils.chat_archive import recent_authors
            from utils import persona_llm
            from utils.output_filter import is_clean

            skip = config.EXCLUDE_USERS | {(self.bot.nick or "").lower()}
            authors = [a for a in recent_authors(channel) if a not in skip]
            directed = self._directed_persona_targets(message.content, authors)
            chance = (
                getattr(config, "REACTION_DIRECTED_CHANCE", 0.0)
                if directed else getattr(config, "REACTION_CHANCE", 0.0)
            )
            if chance <= 0 or random.random() >= chance:
                return
            if time.time() - self._last_reaction.get(channel, 0) < config.REACTION_COOLDOWN:
                return

            targets = directed or authors
            random.shuffle(targets)
            if not getattr(config, "REACTION_USE_LLM", True):
                logging.warning("Persona reactions skipped: reaction_use_llm=false; Markov is command-only.")
                return
            for target in targets[:8]:
                line = await self._persona_line(
                    target, channel, is_clean, persona_llm
                )
                if line:
                    if len(line) > 280:
                        line = line[:279] + "..."
                    self._last_reaction[channel] = time.time()
                    tag = persona_llm.last_model_tag()
                    prefix = f"#{tag} " if tag else ""
                    logging.info(f"Persona reaction in #{channel} as {target}: {line!r}")
                    await message.channel.send(f"🎲 {prefix}{target}: {line}")
                    from utils import reaction_tracker
                    reaction_tracker.watch(channel, line,
                                           {"kind": "ambient", "persona": target,
                                            "model_tag": tag})
                    await self._maybe_continue_reaction(
                        message, target, line, is_clean, persona_llm
                    )
                    return
        except Exception as e:
            logging.warning(f"maybe_react failed: {e}")

    async def handle_regular_message(self, message):
        logging.info(f"Handling regular message from {message.author.name}: {message.content}")

        await self.maybe_react(message)

        user_settings = get_user_settings(message.author.id)
        log_user_activity(message.channel.name, message.author.name)

        # ---- cleaning: strip @mentions, known chatters' names, and emotes ----
        usernames = fetch_usernames(message.channel.name)
        room_id = (getattr(message, 'tags', None) or {}).get('room-id')
        emote_names = ensure_channel_emotes(room_id)

        msg = remove_at_words(message.content)
        msg = filter_usernames_from_message(msg, usernames)
        msg = strip_emotes(msg, emote_names)
        msg = strip_emote_like_tokens(msg)
        msg = msg.strip()

        # nothing meaningful left after cleaning
        if not msg:
            return

        # starts with non-alpha? skip (emotes, punctuation-only, etc.)
        if not msg[0].isalpha():
            return

        # scrub URLs so links are never reposted
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
                # Only output a real, changed translation — never echo the
                # original or post an error string into chat.
                if t and t != msg:
                    mode = user_settings.get('output_mode', 'default')
                    await self.output_message(message, t, mode, user_settings)
            else:
                if is_translation_enabled_globally() and is_translation_enabled_for_channel(message.channel.name):
                    await self.handle_translation_if_needed(message, msg, 'EN')
        else:
            if is_translation_enabled_globally() and is_translation_enabled_for_channel(message.channel.name):
                await self.handle_translation_if_needed(message, msg, 'EN')

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

    async def handle_translation_if_needed(self, message, text, target_language):
        if not self.can_translate or not text:
            return

        target = target_language.upper()
        user_id = message.author.id

        # Confidence across all supported languages. The decision isn't "what's
        # the single top guess" (unreliable when languages are close) but "is
        # this confidently NOT the target language" — i.e. does the best foreign
        # guess clearly beat the target's own score.
        confs = detect_confidences(text)
        if not confs:
            return
        target_conf = confs.get(target, 0.0)
        best_lang, best_conf = None, 0.0
        for code, c in confs.items():
            if code != target and code in config.SUPPORTED_LANGS and c > best_conf:
                best_lang, best_conf = code, c
        if not best_lang:
            return

        # If the target language is the best (or tied) guess, it's the target /
        # too ambiguous -> never translate (this is what stops English & emotes).
        if target_conf >= best_conf:
            return

        # "Confidently foreign" is a question about the *distribution*, not an
        # absolute score: does the best foreign language clearly win the
        # head-to-head against the target? Lingua's confidences are miscalibrated
        # in absolute terms across languages (German tends to score high, Spanish
        # low), so an absolute floor wrongly skipped plain Spanish like "las ratas
        # en mi culo" (ES 0.23) even though English scored a mere 0.05. The
        # reliable signal is the normalized share between the best foreign guess
        # and the target — i.e. of the English-vs-foreign contest, how much does
        # the foreign side win by. A tiny absolute floor still rejects the
        # near-uniform-noise case (everything ~0.05) where share is meaningless.
        denom = best_conf + target_conf
        foreign_share = (best_conf / denom) if denom > 0 else 0.0
        confidently_foreign = (
            best_conf >= config.MIN_FOREIGN_SIGNAL
            and foreign_share >= config.MIN_FOREIGN_SHARE
        )

        # Build the speaker profile from confident foreign detections (kept for
        # the ~speak command and future use; it no longer affects this gate).
        if confidently_foreign:
            record_language(user_id, best_lang)

        if not confidently_foreign:
            return

        # Short-phrase rule: 1-3 word Latin-script messages are unreliable — both
        # lingua and DeepL hallucinate on fragments and on emote names that slip
        # past cleaning ("ge" -> "give", "nah fam" -> "Nah, I'm not hungry"). Only
        # translate a short message when a *single* language is strongly detected
        # (high absolute confidence, e.g. "danke schön" 0.68, "buongiorno" 0.95);
        # ambiguous short text (real or not) is skipped. Non-Latin scripts are
        # exempt — the script itself is a strong signal. This applies to everyone:
        # known-speaker status no longer forces short messages through, which was
        # translating fragments and emote leftovers from established chatters.
        has_non_latin = bool(re.search(r'[^\x00-\x7FÀ-ɏ]', text))
        if (not has_non_latin
                and len(text.split()) < config.MIN_WORDS
                and best_conf < config.MIN_SHORT_CONFIDENCE):
            # ~speak concession: an established speaker of THIS detected
            # language gets a relaxed bar (their short foreign lines are
            # usually real); fragments detecting as some random language
            # still die here — the old blanket bypass translated emote
            # leftovers. This is what makes the stored ~speak flags DO
            # something again.
            relaxed = max(0.0, config.MIN_SHORT_CONFIDENCE - 0.15)
            if not (best_lang in known_languages(user_id) and best_conf >= relaxed):
                return

        txt = self.gtranslate(text, target_language)
        if txt and txt != text:
            logging.info(f"Auto-translate -> #{message.channel.name}: {txt!r}")
            try:
                await message.channel.send(txt)
            except Exception as e:
                logging.error(f"send to #{message.channel.name} FAILED: {e!r}")

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
