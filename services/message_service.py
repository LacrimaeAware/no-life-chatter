# services/message_service.py
import logging
import aiohttp
import asyncio
import random
import time
from collections import deque
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
import unicodedata
from difflib import SequenceMatcher

import config
from utils import translate_optout

# ----------------- helpers -----------------


def _fold_for_compare(s: str) -> str:
    """Case/punctuation/accent-insensitive form for near-copy checks."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))  # drop accents
    s = s.casefold().replace("'", "")
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _near_identical(original: str, translation: str, thresh: float = 0.90) -> bool:
    """True when a 'translation' adds nothing over the original — same text once
    case, punctuation and accents are ignored (typo fixes like rednekc->redneck,
    diacritics like Kysucke->Kysucké, apostrophe/capitalization). Restores the old
    'don't post a near-identical translation' heuristic that regressed to a plain
    exact-string check."""
    a, b = _fold_for_compare(original), _fold_for_compare(translation)
    if not a or not b:
        return True
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= thresh

_RESIDENT_TOPIC_WEAK_TERMS = {
    "about", "actually", "again", "anyone", "before", "being", "better",
    "could", "crap", "doing", "done", "filler", "forced", "good", "gonna",
    "hate", "hates", "just", "made", "maybe", "piece", "play", "really",
    "random", "said", "says", "shit", "some", "someone", "such", "take",
    "thing", "think", "thought", "thoughts", "today", "tonight", "using",
    "wanna", "would",
}

_RESIDENT_TOPIC_SHORT_ANCHORS = {
    "ai", "api", "cs", "egg", "eggs", "gpu", "iq", "lol", "osu", "wow",
}

_RESIDENT_EMOTE_EXACT = None

def _token_set(text):
    return {w.strip("_").lower() for w in re.findall(r"\w+", text or "") if w.strip("_")}


def _raw_tokens_by_lower(text):
    out = {}
    for raw in re.findall(r"\w+", text or ""):
        term = raw.strip("_").lower()
        if term:
            out.setdefault(term, []).append(raw.strip("_"))
    return out


def _resident_exact_emotes():
    global _RESIDENT_EMOTE_EXACT
    if _RESIDENT_EMOTE_EXACT is None:
        try:
            from utils import emote_meaning
            _RESIDENT_EMOTE_EXACT = set(emote_meaning.registry())
        except Exception:
            _RESIDENT_EMOTE_EXACT = set()
    return _RESIDENT_EMOTE_EXACT


def _resident_raw_emote_like(raw):
    if raw in _resident_exact_emotes() and raw.lower() != raw:
        return True
    if re.search(r"[a-z][A-Z]", raw):
        return True
    return len(raw) >= 6 and raw.isupper() and raw.isalpha()


def _resident_topic_anchor(term, raw_tokens):
    if term in _RESIDENT_TOPIC_WEAK_TERMS:
        return False
    if any(_resident_raw_emote_like(raw) for raw in raw_tokens):
        return False
    return len(term) >= 4 or term in _RESIDENT_TOPIC_SHORT_ANCHORS

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
        self._resident_last = {}  # channel -> last resident-persona timestamp
        self._resident_bot_streak = {}  # channel -> bot resident lines since real chat
        self._resident_last_human = {}  # channel -> last non-noise chatter timestamp
        self._resident_recent_authors = {}  # channel -> recent non-noise chatters
        self._resident_idle_checked = {}  # channel -> last idle-roll timestamp
        self._resident_affinity_cache = {}  # (persona, terms) -> (ts, affinity, hits)
        self._resident_idle_task = None
        self._resident_started = time.time()
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

    def _noise_author(self, author):
        name = (author or "").lower()
        return (
            not name
            or name == (self.bot.nick or "").lower()
            or name in getattr(config, "EXCLUDE_USERS", set())
            or name.endswith("bot")
        )

    def _directed_to_resident(self, content, state):
        names = [state.get("persona") or ""]
        bot_name = (self.bot.nick or "").strip()
        if bot_name:
            names.append(bot_name)
        prefix = (state.get("prefix") or "").replace("\U0001f4e3", "").strip()
        if prefix:
            display = prefix.split()[0].strip()
            if len(display) >= 4:
                names.append(display)
        if self._directed_persona_targets(content, names):
            return True
        # extra triggers (e.g. the streamer's name) fire ONLY on an explicit
        # @mention — the bare name is said constantly in that streamer's chat.
        text = (content or "").lower()
        return any(f"@{t}" in text for t in (state.get("triggers") or []) if t)

    def _looks_like_greeting(self, content):
        text = (content or "").lower().strip()
        return bool(re.search(
            r"(^|\b)(gm|good\s+morning|morning|hello|hi|hey|yo|sup|gn|good\s+night)(\b|$)",
            text,
        ))

    def _observe_resident_human(self, message):
        author = message.author.name if message.author else ""
        if self._noise_author(author):
            return
        channel = message.channel.name
        self._resident_last_human[channel] = time.time()
        self._resident_bot_streak[channel] = 0
        authors = self._resident_recent_authors.setdefault(channel, deque(maxlen=20))
        lowered = author.lower()
        try:
            authors.remove(lowered)
        except ValueError:
            pass
        authors.appendleft(lowered)

    def _resident_topic_affinity(self, state, content):
        """Cheap organic-interest signal: did this persona talk about this topic?"""
        from utils import chat_archive

        persona = state.get("persona") or ""
        channel = state.get("channel") or ""
        terms = tuple(chat_archive.query_terms(content, max_terms=8))
        raw_tokens = _raw_tokens_by_lower(content)
        anchors = tuple(term for term in terms if _resident_topic_anchor(term, raw_tokens.get(term, [])))
        if not persona or not anchors:
            return 0.0, 0
        key = (persona, anchors)
        cached = self._resident_affinity_cache.get(key)
        now_ts = time.time()
        if cached and now_ts - cached[0] < 60:
            return cached[1], cached[2]
        hits = chat_archive.search_author_hits(persona, " ".join(anchors), limit=16)
        seen = {}
        same_channel = 0
        covered_terms = set()
        for _id, _sent_at, hit_channel, content in hits:
            line_key = chat_archive.line_match_key(content)
            if not line_key or line_key in seen:
                continue
            hit_terms = _token_set(content)
            covered = set(anchors) & hit_terms
            if not covered:
                continue
            seen[line_key] = True
            covered_terms |= covered
            if chat_archive.normalize_channel(hit_channel) == channel:
                same_channel += 1
        hit_count = len(seen)
        affinity = 0.0
        if hit_count:
            covered_count = len(covered_terms)
            if len(anchors) == 1:
                affinity = min(0.70, max(0, hit_count - 1) / 6.0 + min(same_channel, 3) * 0.03)
            elif covered_count == 1:
                affinity = min(0.45, (hit_count / 10.0) + min(same_channel, 3) * 0.03)
            else:
                affinity = min(
                    1.0,
                    (hit_count / 8.0)
                    + min(covered_count, 3) * 0.12
                    + min(same_channel, 3) * 0.04,
                )
        self._resident_affinity_cache[key] = (now_ts, affinity, hit_count)
        return affinity, hit_count

    def _resident_reply_chance(self, state, message, directed, greeting):
        mode = state.get("mode") or "regular"
        if mode == "response":
            return (float(state.get("directed_chance", 0.65)) if directed else 0.0), 0.0, 0
        if mode == "random":
            return (
                float(state.get("directed_chance", 0.65) if directed else state.get("chance", 0.02)),
                0.0,
                0,
            )
        if directed:
            return float(state.get("directed_chance", 0.65)), 0.0, 0
        if greeting:
            return max(float(state.get("greeting_chance", 0.75)), float(state.get("chance", 0.02))), 0.0, 0

        base = float(state.get("chance", 0.02))
        topic_chance = max(base, float(state.get("topic_chance", 0.16)))
        affinity, hits = self._resident_topic_affinity(state, message.content)
        if affinity <= 0:
            return base, affinity, hits
        curve = max(0.25, float(state.get("topic_curve", 2.0)))
        curved_affinity = affinity ** curve
        return base + (topic_chance - base) * curved_affinity, affinity, hits

    def _resident_prompt(self, state, message, directed, greeting, affinity=0.0, hits=0):
        author = message.author.name if message.author else "someone"
        content = message.content or ""
        context = state.get("context") or ""
        instruction = (
            f"{author} just said in chat: {content}\n"
            "You are temporarily hanging out as this chatter, not answering as an assistant. "
            "Reply only if this chatter would naturally jump in here. "
            "Do not mention being a bot or a persona. Do not include any name label or prefix; "
            "the chat system will add the emote prefix."
        )
        if directed:
            instruction += (
                " This message is directed at you; answer it in-character. "
                "Do not output STOP unless the message is impossible to answer safely."
            )
        else:
            instruction += " If the best move is to stay quiet, output exactly STOP."
        if not directed and greeting:
            instruction += " This looks like a greeting; a short normal greeting back is usually natural."
        elif not directed and affinity >= 0.35:
            instruction += (
                f" This topic appears in your archive ({hits} relevant hits), "
                "so it may be something you would naturally jump into."
            )
        if context:
            instruction += f"\nStanding instruction: {context}"
        return instruction

    def _resident_idle_prompt(self, state):
        channel = state.get("channel") or ""
        context = state.get("context") or ""
        last_human = self._resident_last_human.get(channel, self._resident_started)
        quiet_for = max(0, int(time.time() - last_human))
        recent = list(self._resident_recent_authors.get(channel, []))[:6]
        instruction = (
            f"Chat in #{channel} has been quiet for about {quiet_for // 60} minutes. "
            "You are temporarily hanging out as this chatter, not answering as an assistant. "
            "Write one short natural empty-chat/idle Twitch line only if this chatter would "
            "actually say something into the lull. It can be bored, impatient, observational, "
            "or @ a recently active chatter if that feels natural. If silence is better, output exactly STOP. "
            "Do not mention being a bot or a persona. Do not include any name label or prefix; "
            "the chat system will add the emote prefix."
        )
        if recent:
            instruction += "\nRecently active chatters you may address if natural: " + ", ".join(recent)
        if context:
            instruction += f"\nStanding instruction: {context}"
        return instruction

    async def _send_resident_line(self, channel, line, state, trigger_message=None):
        if trigger_message and state.get("reply_to_trigger", True):
            msg_id = (
                getattr(trigger_message, "id", None)
                or (getattr(trigger_message, "tags", None) or {}).get("id")
            )
            ws = None
            try:
                ws = channel._fetch_websocket()
            except Exception:
                ws = getattr(channel, "_ws", None)
            if msg_id and ws and hasattr(ws, "reply"):
                try:
                    if hasattr(channel, "check_content"):
                        channel.check_content(line)
                    if hasattr(channel, "check_bucket"):
                        channel.check_bucket(channel.name)
                    await ws.reply(msg_id, f"PRIVMSG #{channel.name} :{line}\r\n")
                    return "reply"
                except Exception as e:
                    logging.warning(f"resident reply send failed, falling back to normal send: {e}")
        await channel.send(line)
        return "send"

    def start_resident_idle_loop(self):
        if self._resident_idle_task and not self._resident_idle_task.done():
            return
        self._resident_idle_task = asyncio.create_task(self._resident_idle_loop())

    async def _resident_idle_loop(self):
        await asyncio.sleep(5)
        while True:
            try:
                from utils import resident_persona
                for state in resident_persona.active_channels():
                    await self._maybe_resident_idle(state)
            except Exception as e:
                logging.warning(f"resident idle loop failed: {e}")
            await asyncio.sleep(15)

    async def _maybe_resident_idle(self, state):
        from utils import persona_llm, resident_persona, reaction_tracker
        from utils.output_filter import is_clean

        if state.get("mode") not in {"regular", "random"}:
            return False
        channel_name = state.get("channel")
        channel = self.bot.get_channel(channel_name) if hasattr(self.bot, "get_channel") else None
        if not channel:
            return False
        now_ts = time.time()
        interval = max(10.0, float(state.get("idle_interval", 75.0)))
        if now_ts - self._resident_idle_checked.get(channel_name, 0) < interval:
            return False
        self._resident_idle_checked[channel_name] = now_ts

        idle_after = float(state.get("idle_after", 180.0))
        last_human = self._resident_last_human.get(channel_name, self._resident_started)
        if now_ts - last_human < idle_after:
            return False
        idle_cooldown = float(state.get("idle_cooldown", 240.0))
        if now_ts - self._resident_last.get(channel_name, 0) < idle_cooldown:
            return False
        max_streak = int(state.get("max_bot_streak", 3))
        if self._resident_bot_streak.get(channel_name, 0) >= max_streak:
            return False
        chance = float(state.get("idle_chance", 0.025))
        if chance <= 0 or random.random() >= chance:
            return False

        line = await persona_llm.generate(
            state.get("persona"),
            channel_name,
            self._resident_idle_prompt(state),
            mode="normal",
            copy_strategy="drop",
            invoked_by="resident-idle",
            candidates=1,
        )
        if not line or is_stop_followup(line):
            return False
        line = resident_persona.format_line(state, line)
        if not line or not is_clean(line):
            return False
        if len(line) > 450:
            line = line[:449] + "..."
        self._resident_last[channel_name] = now_ts
        self._resident_bot_streak[channel_name] = self._resident_bot_streak.get(channel_name, 0) + 1
        logging.info(f"Resident idle in #{channel_name} as {state.get('persona')}: {line!r}")
        await self._send_resident_line(channel, line, state)
        reaction_tracker.watch(channel_name, line, {
            "kind": "resident_idle",
            "persona": state.get("persona"),
            "mode": state.get("mode"),
        })
        return True

    async def maybe_resident_react(self, message):
        """Channel-scoped resident persona mode controlled by live state."""
        channel = message.channel.name
        try:
            from utils import persona_llm, resident_persona
            from utils.chat_archive import normalize_author
            from utils.output_filter import is_clean
            from utils import reaction_tracker

            state = resident_persona.get(channel)
            if not state or state.get("mode") == "silent":
                return False

            author = message.author.name if message.author else ""
            if self._noise_author(author):
                return False

            persona = state.get("persona")
            if normalize_author(author) == normalize_author(persona):
                return False

            mode = state.get("mode") or "regular"
            directed = self._directed_to_resident(message.content, state)
            greeting = self._looks_like_greeting(message.content)
            chance, affinity, hits = self._resident_reply_chance(state, message, directed, greeting)
            if chance <= 0 or random.random() >= chance:
                return False

            cooldown = float(
                state.get("directed_cooldown", 0.0)
                if directed else state.get("cooldown", 180.0)
            )
            if time.time() - self._resident_last.get(channel, 0) < cooldown:
                return False

            prompt = self._resident_prompt(state, message, directed, greeting, affinity, hits)
            line = await persona_llm.generate(
                persona,
                channel,
                prompt,
                mode="normal",
                copy_strategy="drop",
                invoked_by=author,
                candidates=1,
            )
            if not line or is_stop_followup(line):
                return False
            line = resident_persona.format_line(state, line)
            if not line or not is_clean(line):
                return False
            if len(line) > 450:
                line = line[:449] + "..."
            self._resident_last[channel] = time.time()
            self._resident_bot_streak[channel] = self._resident_bot_streak.get(channel, 0) + 1
            sent_as = await self._send_resident_line(message.channel, line, state, trigger_message=message)
            logging.info(
                f"Resident persona in #{channel} as {persona} ({sent_as}, "
                f"chance={chance:.3g}, affinity={affinity:.2g}, hits={hits}): {line!r}"
            )
            reaction_tracker.watch(channel, line, {
                "kind": "resident",
                "persona": persona,
                "mode": mode,
                "directed": directed,
                "affinity": affinity,
                "hits": hits,
            })
            return True
        except Exception as e:
            logging.warning(f"maybe_resident_react failed: {e}")
            return False

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

        self._observe_resident_human(message)
        resident_responded = await self.maybe_resident_react(message)
        if not resident_responded:
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
                # Only output a real, *meaningfully different* translation — never
                # echo the original, a near-identical typo/diacritic fix, or an
                # error string.
                if t and not _near_identical(msg, t):
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
        # super-admin opt-out: never auto-translate these users (they write
        # English slang the detector misreads as foreign). ~notranslate.
        if message.author and translate_optout.is_opted_out(message.author.name):
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
        if txt and not _near_identical(text, txt):
            logging.info(f"Auto-translate -> #{message.channel.name}: {txt!r}")
            try:
                await message.channel.send(txt)
            except Exception as e:
                logging.error(f"send to #{message.channel.name} FAILED: {e!r}")
        elif txt:
            logging.info(f"Auto-translate suppressed (near-identical): {txt!r}")

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
