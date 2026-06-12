"""Ground-truth emote registry with ALIASES, plus a suspect-emote oracle queue.

7TV emote-set entries carry both the in-chat alias (what people type — e.g. a
friend's name) and the ORIGINAL emote name, plus image URLs. That triple is
the key to 'double awareness': an aliased emote means its alias-word AND its
image at once, and the clash between them is usually the joke.

Builds data/unsynced/emote_registry.json:
    {token: {"origin": "7tv|bttv|ffz", "channel": ..., "original": ...,
             "image": ...}}
for the configured channels + any --channels extras. Then emits a review
queue of SUSPECTED emotes — emote-shaped tokens common in the archive but
absent from every known set (could be dead emotes, fake personal emotes, or
real emotes from unfetched channels) — for the user to verify, which becomes
training data for a learned emote detector.

    python scripts/build_emote_registry.py [--channels a,b,c] [--suspects 40]
"""

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from utils import chat_archive, persona_classifier as pc  # noqa: E402

OUT = os.path.join("data", "unsynced", "emote_registry.json")
DROPOFF = os.path.join("..", "ai-prompt-engineering", "dropoff")


def _json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "NoLifeChatter"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def room_id(login):
    tok = json.load(open(config.TOKEN_FILE))["access_token"]
    tok = tok[6:] if tok.startswith("oauth:") else tok
    d = _json(f"https://api.twitch.tv/helix/users?login={login}",
              {"Authorization": f"Bearer {tok}", "Client-Id": config.TWITCH_CLIENT_ID})
    return d["data"][0]["id"] if d.get("data") else None


def fetch_channel(login, rid, reg):
    def put(token, origin, original=None, image=None):
        if token and token not in reg:
            reg[token] = {"origin": origin, "channel": login,
                          "original": original or token, "image": image}
    try:  # 7TV (aliases live here: entry name vs data.name)
        j = _json(f"https://7tv.io/v3/users/twitch/{rid}")
        for e in ((j.get("emote_set") or {}).get("emotes") or []):
            host = ((e.get("data") or {}).get("host") or {}).get("url", "")
            img = f"https:{host}/2x.webp" if host else None
            put(e.get("name"), "7tv", (e.get("data") or {}).get("name"), img)
    except Exception as ex:
        print(f"  7tv {login}: {ex}")
    try:  # BTTV
        j = _json(f"https://api.betterttv.net/3/cached/users/twitch/{rid}")
        for e in (j.get("channelEmotes") or []) + (j.get("sharedEmotes") or []):
            put(e.get("code"), "bttv", e.get("code"),
                f"https://cdn.betterttv.net/emote/{e.get('id')}/2x")
    except Exception as ex:
        print(f"  bttv {login}: {ex}")
    try:  # FFZ
        j = _json(f"https://api.frankerfacez.com/v1/room/id/{rid}")
        for s in (j.get("sets") or {}).values():
            for e in (s.get("emoticons") or []):
                urls = e.get("urls") or {}
                put(e.get("name"), "ffz", e.get("name"),
                    list(urls.values())[-1] if urls else None)
    except Exception as ex:
        print(f"  ffz {login}: {ex}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels", default="",
                    help="extra channels beyond config bot.channels")
    ap.add_argument("--suspects", type=int, default=40)
    args = ap.parse_args()
    channels = list(dict.fromkeys(
        [c.lower() for c in config.CHANNELS]
        + [c.strip().lower() for c in args.channels.split(",") if c.strip()]))
    reg = {}
    if os.path.exists(OUT):
        reg = json.load(open(OUT, encoding="utf-8"))
    for login in channels:
        rid = room_id(login)
        if not rid:
            print(f"  {login}: no room id")
            continue
        before = len(reg)
        fetch_channel(login, rid, reg)
        print(f"  {login}: +{len(reg) - before} emotes")
    # globals
    try:
        j = _json("https://7tv.io/v3/emote-sets/global")
        for e in (j.get("emotes") or []):
            if e.get("name") and e["name"] not in reg:
                reg[e["name"]] = {"origin": "7tv-global", "channel": None,
                                  "original": e["name"], "image": None}
    except Exception:
        pass
    json.dump(reg, open(OUT, "w", encoding="utf-8"), indent=0)
    aliased = [(t, d["original"]) for t, d in reg.items()
               if d.get("original") and d["original"] != t]
    print(f"\nregistry: {len(reg)} emotes ({len(aliased)} ALIASED) -> {OUT}")
    for t, o in aliased[:10]:
        print(f"  alias: {t} -> {o}")

    # suspect-emote queue: emote-shaped, used by multiple people, NOT in registry
    model = pc.load()
    from collections import Counter
    usage = Counter()
    for prof in (model.get("profiles") or {}).values():
        for e in prof.get("emotes", {}):
            usage[e] += 1
    known = {k.lower() for k in reg}
    suspects = [(e, n) for e, n in usage.most_common()
                if e.lower() not in known and n >= 2][:args.suspects]
    qdir = os.path.join(DROPOFF, "nolifechatter_emote_suspects_v1")
    os.makedirs(os.path.join(qdir, "results"), exist_ok=True)
    with open(os.path.join(qdir, "queue.jsonl"), "w", encoding="utf-8") as fh:
        for rank, (e, n) in enumerate(suspects):
            conn = chat_archive.connect()
            try:
                ex = conn.execute(
                    "SELECT m.content FROM messages_fts f JOIN messages m ON m.id=f.rowid "
                    "WHERE f.messages_fts MATCH ? LIMIT 2", (f'"{e}"',)).fetchall()
            except Exception:
                ex = []
            fh.write(json.dumps({
                "id": f"nlc-emote-{rank:04d}", "source": "NoLifeChatter",
                "kind": "single-classification",
                "question": "Is this token an EMOTE (vs a word/name/typo)?",
                "subject": {"token": e, "used_by_n_people": n,
                            "example_messages": [x[0][:120] for x in ex]},
                "options": ["emote", "not an emote", "alias of another emote",
                            "i don't know"],
                "allow_other": True,
                "answer": None, "answer_note": None, "answered_at": None,
            }, ensure_ascii=False) + "\n")
    print(f"suspect queue: {len(suspects)} items -> {qdir}/queue.jsonl")


if __name__ == "__main__":
    main()
