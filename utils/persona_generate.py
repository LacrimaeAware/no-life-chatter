"""~generate: tag-driven example generation (docs/GENERATE_AND_BOT_MODES.md).

A recipe is an unordered bag of tags — chatter names (one = their voice,
several = a fusion), trait poles (maximally optimist/doomer/...), channel and
year scopes, a free-text topic, and engine/model picks. Saved per-user combos
expand into their tags (one level deep).

LLM recipes are prompt-built from each chatter's (scoped) real messages plus
trait register hints; pure-trait recipes are prompting only. Markov recipes
recombine the chatters' own words (no traits/topic possible there).
"""

import random
import re
import sqlite3

import config
from services import llm
from utils import chat_archive, persona_markov
from utils.persona_traits import AXES, pole_map


# ----------------------------- saved combos -----------------------------

def _conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gen_combos ("
        " user_id TEXT NOT NULL, name TEXT NOT NULL, tags TEXT NOT NULL,"
        " PRIMARY KEY (user_id, name))"
    )
    return conn


def save_combo(user, name, tags):
    name = name.lower()
    if name in RESERVED or name in pole_map():
        return f"'{name}' is reserved — pick another name."
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO gen_combos (user_id, name, tags) VALUES (?,?,?)",
            (user.lower(), name, " ".join(tags)),
        )
    return None


def get_combo(user, name):
    with _conn() as conn:
        row = conn.execute(
            "SELECT tags FROM gen_combos WHERE user_id=? AND name=?",
            (user.lower(), name.lower()),
        ).fetchone()
    return row[0].split() if row else None


def list_combos(user):
    with _conn() as conn:
        return [(n, t) for n, t in conn.execute(
            "SELECT name, tags FROM gen_combos WHERE user_id=? ORDER BY name",
            (user.lower(),),
        )]


def delete_combo(user, name):
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM gen_combos WHERE user_id=? AND name=?",
            (user.lower(), name.lower()),
        )
    return cur.rowcount > 0


RESERVED = {"save", "list", "del", "delete", "help"}


# ----------------------------- recipe parsing -----------------------------

def parse_recipe(params, invoking_user):
    """Bag-of-tags -> recipe dict. Unrecognized words become the topic.
    Saved combos (the invoking user's) expand one level."""
    from utils.persona_classifier import _known_usernames
    users_set = _known_usernames()
    poles = pole_map()
    recipe = {"users": [], "traits": [], "channels": [], "year": None,
              "topic": [], "engine": "llm", "model": None, "expanded": []}

    # commas are separators, not syntax
    tokens = []
    for p in params:
        tokens.extend(t for t in p.replace(",", " ").split() if t)

    def absorb(tok, allow_combo=True):
        low = tok.lower().lstrip("@")
        if low.startswith("chat="):
            v = low.split("=", 1)[1].strip("#")
            if v:
                recipe["channels"].append(v)
            return
        if low.startswith("year=") and low.split("=", 1)[1].isdigit():
            recipe["year"] = int(low.split("=", 1)[1])
            return
        if low.startswith("engine="):
            recipe["engine"] = low.split("=", 1)[1]
            return
        if low.startswith("model="):
            from utils.persona_llm import resolve_model
            recipe["model"] = resolve_model(low.split("=", 1)[1])
            return
        if low.startswith("topic="):
            v = low.split("=", 1)[1]
            if v:
                recipe["topic"].append(v)
            return
        if low in poles:
            if low not in recipe["traits"]:
                recipe["traits"].append(low)
            return
        canon = chat_archive.normalize_author(low)
        if canon in users_set:
            if canon not in recipe["users"]:
                recipe["users"].append(canon)
            return
        if allow_combo:
            stored = get_combo(invoking_user, low)
            if stored is not None:
                recipe["expanded"].append(low)
                for t in stored:
                    absorb(t, allow_combo=False)  # combos can't nest combos
                return
        recipe["topic"].append(tok)  # leftover words = topic free-text

    for tok in tokens:
        absorb(tok)
    recipe["topic"] = " ".join(recipe["topic"]).strip()
    return recipe


# ----------------------------- generation -----------------------------

_BAD_OUTPUT_RE = re.compile(r"^[~$!][A-Za-z]{2,}")


def _scoped_messages(author, channels, year):
    if channels:
        msgs = []
        for ch in channels:
            msgs.extend(chat_archive.messages_for(author, channel=ch, year=year))
        return msgs
    return chat_archive.messages_for(author, year=year)


def _exemplars_for(author, channels, year, n):
    from utils.persona_llm import _usable_exemplar
    pool = [m for m in _scoped_messages(author, channels, year) if _usable_exemplar(m)]
    random.shuffle(pool)  # fresh sample every call — ~generate wants variety
    return pool[:n]


def _ok(text):
    if not text or len(text) < 2 or len(text) > 480:
        return False
    if "http://" in text or "https://" in text or "www." in text:
        return False
    return not _BAD_OUTPUT_RE.match(text.lstrip())


async def generate_example(recipe):
    """One example message from a recipe, or (None, reason)."""
    users, traits = recipe["users"], recipe["traits"]
    if recipe["engine"] == "markov":
        if not users:
            return None, "engine=markov needs at least one chatter tag"
        msgs = []
        for u in users:
            msgs.extend(_scoped_messages(u, recipe["channels"], recipe["year"]))
        model = persona_markov.build_from_messages(msgs, label="+".join(users))
        if not model:
            return None, "not enough archived messages for that recipe"
        for _ in range(6):
            out = persona_markov.generate(model)
            if out and _ok(out):
                return out, None
        return None, "markov produced nothing usable"

    if not users and not traits and not recipe["topic"]:
        return None, "give me something — chatters, a trait, or a topic"

    parts = ["You write ONE single Twitch chat message as an example. "
             "Output only the message — no quotes, no explanation."]
    if users:
        if len(users) > 1:
            parts.append(
                f"Write it as a FUSION of the chatters {', '.join(users)} — one "
                f"voice blending their vocabularies, emotes, energy and habits.")
        else:
            parts.append(f"Write it in the voice of the chatter {users[0]}.")
        budget = max(30, 120 // len(users))
        for u in users:
            ex = _exemplars_for(u, recipe["channels"], recipe["year"], budget)
            if ex:
                parts.append(f"Real messages by {u}:\n" + "\n".join(ex))
    if traits:
        for t in traits:
            axis, sign = pole_map()[t]
            neg, pos, neg_s, pos_s = AXES[axis]
            hints = pos_s if sign > 0 else neg_s
            parts.append(
                f"The message must read MAXIMALLY {t}. Register examples "
                f"(do not copy them): " + " | ".join(hints[:3]))
    if recipe["topic"]:
        parts.append(f"The message is about: {recipe['topic']}.")
    parts.append("Make it fresh and specific — not generic, not a rephrasing "
                 "of any example above.")
    messages = [{"role": "system", "content": "\n\n".join(parts)},
                {"role": "user", "content": "Write the message now."}]

    for _ in range(2):
        raw = await llm.chat(messages, max_tokens=150, temperature=1.0,
                             model=recipe["model"])
        if raw:
            out = raw.strip().strip('"')
            if _ok(out):
                return out, None
    return None, llm.last_error() or "model returned nothing usable"
