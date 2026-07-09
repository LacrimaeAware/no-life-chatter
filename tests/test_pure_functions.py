import importlib
import math
import sys
import threading
import types
import unittest
from collections import Counter


def install_fake_config():
    fake = types.SimpleNamespace(
        PREFIX="~",
        ARCHIVE_ALIASES={"legacyroom": "mainroom"},
        ARCHIVE_USER_ALIASES={"altone": "mainuser", "oldalt": "altone"},
        ARCHIVE_CHANNEL_ALIASES={"side-room": "mainroom"},
        ARCHIVE_DB="data/unsynced/test_archive.db",
        CLASSIFIER_FILE="data/unsynced/test_classifier.pkl",
        EXCLUDE_USERS={"helperbot"},
        LLM_SEMANTIC_MIN_SCORE=0.50,
        LLM_SEMANTIC_UNANCHORED_MIN_SCORE=0.62,
    )
    sys.modules["config"] = fake
    return fake


install_fake_config()
sys.modules["services.llm"] = types.SimpleNamespace(chat=None)
from utils import archive_qa, chat_archive, emote_explain, fact_bank, message_quality, persona_classifier, persona_iq, persona_llm, resident_persona  # noqa: E402
from commands import markers  # noqa: E402


class ArchiveNormalizationTests(unittest.TestCase):
    def test_line_match_key_normalizes_punctuation_case_and_spacing(self):
        left = "You just dobray'd your last hand-no buddy!"
        right = "you just dobray\u2019d your last hand no buddy"
        self.assertEqual(chat_archive.line_match_key(left), chat_archive.line_match_key(right))

    def test_line_similarity_detects_close_substring_copy(self):
        short = "this is the exact copied phrase with enough length"
        long = "@someone this is the exact copied phrase with enough length lol"
        self.assertGreaterEqual(chat_archive.line_similarity(short, long), 0.97)

    def test_line_similarity_returns_zero_for_empty_normalized_text(self):
        self.assertEqual(chat_archive.line_similarity("!!!", "???"), 0.0)

    def test_author_alias_chains_and_author_keys(self):
        self.assertEqual(chat_archive.normalize_author("@OldAlt,"), "mainuser")
        self.assertEqual(chat_archive.author_keys("oldalt"), ["altone", "mainuser", "oldalt"])

    def test_archive_connections_are_thread_local(self):
        main_conn = chat_archive.connect()
        seen = []

        def worker():
            seen.append(chat_archive.connect())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertIsNot(main_conn, seen[0])

    def test_alias_cycle_stops_instead_of_looping_forever(self):
        aliases = {"a": "b", "b": "a"}
        self.assertIn(chat_archive._resolve_alias("a", aliases), {"a", "b"})

    def test_query_terms_drop_basic_scaffolding_words(self):
        terms = chat_archive.query_terms("thats such an answer and the wrong one about wow")
        self.assertNotIn("thats", terms)
        self.assertNotIn("answer", terms)
        self.assertNotIn("and", terms)
        self.assertNotIn("see", terms)
        self.assertNotIn("the", terms)
        self.assertNotIn("well", terms)
        self.assertIn("wow", terms)

    def test_context_window_does_not_invent_context_from_author_only_source(self):
        conn = chat_archive.connect()
        channel = "ctx_author_only"
        with conn:
            conn.execute("DELETE FROM messages WHERE channel = ?", (channel,))
            conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
                "VALUES (?, ?, ?, ?, 'zonian', ?)",
                (channel, "mainuser", "2026-01-01 00:00:01", "before fragment",
                 "data/unsynced/external_logs/zonian/raw/room/mainuser/2026-01.log"),
            )
            cur = conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
                "VALUES (?, ?, ?, ?, 'zonian', ?)",
                (channel, "mainuser", "2026-01-01 00:00:02", "target fragment",
                 "data/unsynced/external_logs/zonian/raw/room/mainuser/2026-01.log"),
            )
            hit_id = cur.lastrowid
            conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
                "VALUES (?, ?, ?, ?, 'zonian', ?)",
                (channel, "mainuser", "2026-01-01 00:00:03", "after fragment",
                 "data/unsynced/external_logs/zonian/raw/room/mainuser/2026-01.log"),
            )

        window = chat_archive.context_window(hit_id, channel, before=2, after=2)
        self.assertEqual([(row[0], row[2]) for row in window], [(hit_id, "target fragment")])

    def test_context_window_allows_zonian_when_other_speakers_are_nearby(self):
        conn = chat_archive.connect()
        channel = "ctx_multi_author"
        with conn:
            conn.execute("DELETE FROM messages WHERE channel = ?", (channel,))
            conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
                "VALUES (?, ?, ?, ?, 'zonian', ?)",
                (channel, "otheruser", "2026-01-01 00:00:01", "real previous chat",
                 "data/unsynced/external_logs/zonian/raw/room/otheruser/2026-01.log"),
            )
            cur = conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
                "VALUES (?, ?, ?, ?, 'zonian', ?)",
                (channel, "mainuser", "2026-01-01 00:00:02", "target reply",
                 "data/unsynced/external_logs/zonian/raw/room/mainuser/2026-01.log"),
            )
            hit_id = cur.lastrowid
            conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
                "VALUES (?, ?, ?, ?, 'zonian', ?)",
                (channel, "thirduser", "2026-01-01 00:00:03", "real next chat",
                 "data/unsynced/external_logs/zonian/raw/room/thirduser/2026-01.log"),
            )

        window = chat_archive.context_window(hit_id, channel, before=2, after=2)
        self.assertEqual([row[2] for row in window],
                         ["real previous chat", "target reply", "real next chat"])

    def test_context_window_dedupes_alias_repeated_lines(self):
        conn = chat_archive.connect()
        channel = "ctx_alias_dupes"
        with conn:
            conn.execute("DELETE FROM messages WHERE channel = ?", (channel,))
            conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
                "VALUES (?, ?, ?, ?, 'chatterino', ?)",
                (channel, "oldalt", "2026-01-01 00:00:01", "same copied line",
                 "C:/logs/ctx_alias_dupes-2026-01-01.log"),
            )
            cur = conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
                "VALUES (?, ?, ?, ?, 'chatterino', ?)",
                (channel, "mainuser", "2026-01-01 00:00:02", "same copied line",
                 "C:/logs/ctx_alias_dupes-2026-01-01.log"),
            )
            hit_id = cur.lastrowid
            conn.execute(
                "INSERT INTO messages (channel, author, sent_at, content, source, src_path) "
                "VALUES (?, ?, ?, ?, 'chatterino', ?)",
                (channel, "otheruser", "2026-01-01 00:00:03", "different line",
                 "C:/logs/ctx_alias_dupes-2026-01-01.log"),
            )

        window = chat_archive.context_window(hit_id, channel, before=2, after=2)
        self.assertEqual([row[2] for row in window], ["same copied line", "different line"])
        self.assertEqual(window[0][0], hit_id)

    def test_exact_search_dedupes_same_line_within_one_minute(self):
        conn = chat_archive.connect()
        channel = "search_dedupe_room"
        phrase = "same exact question"
        line = "same exact question @MainUser"
        with conn:
            conn.execute("DELETE FROM messages WHERE channel = ?", (channel,))
            for sent_at in (
                "2026-01-01 00:00:01",
                "2026-01-01 00:00:04",
                "2026-01-01 00:01:04",
            ):
                conn.execute(
                    "INSERT INTO messages (channel, author, sent_at, content, source) "
                    "VALUES (?, ?, ?, ?, 'test')",
                    (channel, "searchperson", sent_at, line),
                )

        self.assertEqual(chat_archive.search_all_count(phrase, channel=channel), 2)
        self.assertEqual(len(chat_archive.search_all(phrase, limit=5, channel=channel)), 2)
        total, rows = chat_archive.said("searchperson", phrase, limit=5, channel=channel)
        self.assertEqual(total, 2)
        self.assertEqual(len(rows), 2)


class ScopeParserTests(unittest.TestCase):
    def test_parse_scope_defaults_to_current_channel(self):
        rest, channel, year = markers._parse_scope(["person"], "mainroom")
        self.assertEqual(rest, ["person"])
        self.assertEqual(channel, "mainroom")
        self.assertIsNone(year)

    def test_parse_scope_accepts_all_channel_and_year(self):
        rest, channel, year = markers._parse_scope(
            ["person", "chat=all", "year=2025"], "mainroom")
        self.assertEqual(rest, ["person"])
        self.assertIsNone(channel)
        self.assertEqual(year, 2025)

    def test_parse_scope_strips_channel_hash(self):
        rest, channel, year = markers._parse_scope(["chat=#OtherRoom", "person"], "mainroom")
        self.assertEqual(rest, ["person"])
        self.assertEqual(channel, "otherroom")
        self.assertIsNone(year)


class PersonaClassifierPureTests(unittest.TestCase):
    def test_is_emote_token_detects_case_marked_emotes(self):
        self.assertTrue(persona_classifier._is_emote_token("FeelsOkayMan"))
        self.assertTrue(persona_classifier._is_emote_token("OMEGALUL"))
        self.assertFalse(persona_classifier._is_emote_token("regularword"))
        self.assertFalse(persona_classifier._is_emote_token("word123"))

    def test_strip_emote_tokens_removes_only_emote_shaped_tokens(self):
        text = "hello FeelsOkayMan regularword OMEGALUL"
        self.assertEqual(persona_classifier.strip_emote_tokens(text), "hello regularword")

    def test_logodds_profile_orders_distinctive_terms(self):
        author = Counter({"signature": 8, "shared": 4, "tiny": 1})
        background = Counter({"shared": 40, "background": 80})
        profile = persona_classifier._logodds_profile(
            author, sum(author.values()), background, sum(background.values()),
            top=2, min_count=3)
        self.assertIn("signature", profile)
        self.assertNotIn("tiny", profile)
        norm = math.sqrt(sum(weight * weight for weight in profile.values()))
        self.assertAlmostEqual(norm, 1.0)

    def test_logodds_profile_excludes_common_panel_terms(self):
        author = Counter({"signature": 8, "everyone": 8})
        background = Counter({"everyone": 2})
        profile = persona_classifier._logodds_profile(
            author, sum(author.values()), background, sum(background.values()),
            top=5, min_count=3,
            prevalence={"signature": 1, "everyone": 9}, n_panel=10)
        self.assertIn("signature", profile)
        self.assertNotIn("everyone", profile)


class MessageQualityPureTests(unittest.TestCase):
    def test_rejects_command_and_bot_syntax(self):
        self.assertFalse(message_quality.usable_for_persona_exemplar("$gpt tell me a thing"))
        self.assertFalse(message_quality.usable_for_persona_exemplar("<groq what is this"))
        self.assertFalse(message_quality.usable_for_iq("^guess a ^guess b ^guess c"))
        self.assertFalse(message_quality.usable_for_iq("[Translation] I am not allowed to translate that"))

    def test_rejects_repeated_emote_or_token_spam(self):
        text = "pepeLaugh pepeLaugh pepeLaugh pepeLaugh pepeLaugh pepeLaugh"
        self.assertFalse(message_quality.usable_for_iq(text))
        phrase = "because the model copied the same clause " * 4
        self.assertFalse(message_quality.usable_for_iq(phrase))

    def test_collapses_repeated_spans(self):
        text = "i mean this because it works i mean this because it works"
        self.assertEqual(message_quality.clean_text(text), "i mean this because it works")

    def test_keeps_reasonable_semantic_text(self):
        text = "because jupyter needs the kernel packages installed for interactive python"
        self.assertTrue(message_quality.usable_for_iq(text))
        self.assertIsNotNone(message_quality.semantic_text(text))


class FactBankPureTests(unittest.TestCase):
    def test_extracts_self_claims_as_candidates_not_truths(self):
        rows = fact_bank.extract_claims("OldAlt", "I'm a software guy and I love graph theory lol")
        claims = {(row["kind"], row["claim"]) for row in rows}
        self.assertIn(("self_identity", "a software guy"), claims)
        self.assertIn(("preference_positive", "graph theory"), claims)
        self.assertTrue(all(row["author"] == "mainuser" for row in rows))

    def test_rejects_questions_and_commands(self):
        self.assertEqual(fact_bank.extract_claims("mainuser", "am I a software guy?"), [])
        self.assertEqual(fact_bank.extract_claims("mainuser", "~persona me I am smart"), [])

    def test_rejects_overbroad_possession_capture(self):
        rows = fact_bank.extract_claims(
            "mainuser",
            "I almost pissed my whole bed dreaming i was pissing in a dream",
        )
        self.assertFalse(any(row["kind"] == "possession" for row in rows))


class ArchiveQaPureTests(unittest.TestCase):
    def test_parse_params_accepts_author_and_chat_scope(self):
        parsed = archive_qa.parse_params(
            ["user=OldAlt", "chat=here", "graph", "theory"],
            current_channel="Side-Room",
        )
        self.assertEqual(parsed["author"], "mainuser")
        self.assertEqual(parsed["channel"], "mainroom")
        self.assertEqual(parsed["query"], "graph theory")

    def test_parse_params_strips_repeated_scoped_author_from_query(self):
        parsed = archive_qa.parse_params(
            ["user=OldAlt", "OldAlt", "loves", "graph", "theory"],
            current_channel="Side-Room",
        )
        self.assertEqual(parsed["author"], "mainuser")
        self.assertEqual(parsed["query"], "loves graph theory")

    def test_author_hits_dedupes_duplicate_imported_lines(self):
        seen = []
        rows = [
            (1, "2026-01-01 00:00:01", "mainroom", "I love graph theory"),
            (2, "2026-01-01 00:00:02", "mainroom", "i love graph theory!!"),
            (3, "2026-01-01 00:00:03", "mainroom", "I love number theory"),
        ]
        original = chat_archive.search_author_hits
        try:
            chat_archive.search_author_hits = lambda *args, **kwargs: rows
            hits = archive_qa._author_hits("mainuser", "graph theory", limit=5)
            seen.extend(hit["text"] for hit in hits)
        finally:
            chat_archive.search_author_hits = original
        self.assertEqual(seen, ["I love graph theory", "I love number theory"])

    def test_author_hits_filters_preference_verb_without_topic(self):
        rows = [
            (1, "2026-01-01 00:00:01", "mainroom", "someone loves papaplatte"),
            (2, "2026-01-01 00:00:02", "mainroom", "deviled eggs Evilge"),
        ]
        original = chat_archive.search_author_hits
        try:
            chat_archive.search_author_hits = lambda *args, **kwargs: rows
            hits = archive_qa._author_hits("mainuser", "loves eggs", limit=5)
        finally:
            chat_archive.search_author_hits = original
        self.assertEqual([hit["text"] for hit in hits], ["deviled eggs Evilge"])

    def test_format_chat_returns_compact_evidence(self):
        report = {
            "query": "graph theory",
            "author": "mainuser",
            "channel": None,
            "terms": ["graph", "theory"],
            "facts": [{
                "kind": "preference_positive",
                "claim": "graph theory",
                "support_count": 2,
                "sent_at": "2026-01-01 00:00:00",
            }],
            "archive": [],
            "near": [],
            "emotes": [],
        }
        out = archive_qa.format_chat(report, max_chars=180)
        self.assertIn("mainuser", out)
        self.assertIn("graph theory", out)
        self.assertLessEqual(len(out), 180)

    def test_format_chat_suppresses_weak_one_off_claims(self):
        report = {
            "query": "math",
            "author": "mainuser",
            "channel": None,
            "terms": ["math"],
            "facts": [{
                "kind": "self_identity",
                "claim": "on drugs (math) and pasted model output",
                "support_count": 1,
                "sent_at": "2026-01-01 00:00:00",
            }],
            "archive": [],
            "near": [],
            "emotes": [],
        }
        out = archive_qa.format_chat(report, max_chars=180)
        self.assertIn("No strong archive evidence", out)
        self.assertNotIn("on drugs", out)

    def test_answer_prompt_uses_receipts_not_weak_claims(self):
        report = {
            "query": "graph theory",
            "author": "mainuser",
            "channel": None,
            "terms": ["graph", "theory"],
            "facts": [{
                "kind": "self_identity",
                "claim": "a weird one-off regex claim",
                "support_count": 1,
                "sent_at": "2026-01-01 00:00:00",
            }],
            "archive": [{
                "author": "mainuser",
                "channel": "mainroom",
                "sent_at": "2026-01-01 00:00:00",
                "text": "I love graph theory",
            }],
            "near": [],
            "emotes": [],
        }
        messages = archive_qa.answer_messages(report)
        joined = "\n".join(message["content"] for message in messages)
        self.assertIn("[A1]", joined)
        self.assertIn("I love graph theory", joined)
        self.assertNotIn("weird one-off", joined)

    def test_format_answer_appends_labels_when_model_omits_them(self):
        report = {
            "query": "graph theory",
            "author": "mainuser",
            "channel": None,
            "terms": ["graph", "theory"],
            "facts": [],
            "archive": [{
                "author": "mainuser",
                "channel": "mainroom",
                "sent_at": "2026-01-01 00:00:00",
                "text": "I love graph theory",
            }],
            "near": [],
            "emotes": [],
        }
        out = archive_qa.format_answer_chat(report, "They directly said it.")
        self.assertIn("[A1]", out)

    def test_opinion_is_query_scaffolding_not_topic(self):
        terms = chat_archive.query_terms("opinion on his country")
        self.assertEqual(terms, ["country"])

    def test_askchat_emote_hits_ignore_plain_lowercase_words(self):
        original_registry = archive_qa.emote_meaning.registry
        original_lookup = archive_qa.emote_meaning.lookup
        original_nearest = archive_qa.emote_meaning.nearest_emotes
        try:
            archive_qa.emote_meaning.registry = lambda: {"his": {"tags": ["fake"]}}
            archive_qa.emote_meaning.lookup = lambda raw: (raw.upper(), {"tags": ["fake"]})
            archive_qa.emote_meaning.nearest_emotes = lambda raw, n=4: [("HIS", 0.9)]
            self.assertEqual(archive_qa._emote_hits("his country"), [])
        finally:
            archive_qa.emote_meaning.registry = original_registry
            archive_qa.emote_meaning.lookup = original_lookup
            archive_qa.emote_meaning.nearest_emotes = original_nearest

    def test_evidence_items_filter_unanchored_archive_noise(self):
        report = {
            "query": "opinion on his country",
            "author": "mainuser",
            "channel": None,
            "terms": ["country"],
            "facts": [],
            "archive": [
                {
                    "author": "mainuser",
                    "channel": "mainroom",
                    "sent_at": "2026-01-01 00:00:00",
                    "text": "opinions on the mass shooting in the US",
                },
                {
                    "author": "mainuser",
                    "channel": "mainroom",
                    "sent_at": "2026-01-01 00:00:01",
                    "text": "invaded the wrong country",
                },
            ],
            "near": [],
            "emotes": [],
        }
        evidence = archive_qa.evidence_items(report)
        joined = " ".join(item["text"] for item in evidence)
        self.assertIn("wrong country", joined)
        self.assertNotIn("mass shooting", joined)

    def test_fact_hits_filter_single_term_unrelated_claims(self):
        rows = [
            {
                "author": "mainuser",
                "kind": "self_identity",
                "claim": "running ai for my game",
                "support_count": 2,
                "confidence": 0.8,
                "evidence": [{"clean_text": "running ai for my game"}],
            },
            {
                "author": "mainuser",
                "kind": "belief",
                "claim": "all women are evil",
                "support_count": 2,
                "confidence": 0.8,
                "evidence": [{"clean_text": "all women are evil"}],
            },
        ]
        original_load = archive_qa.fact_bank.load_jsonl
        original_search = archive_qa.fact_bank.search
        try:
            archive_qa.fact_bank.load_jsonl = lambda: rows
            archive_qa.fact_bank.search = lambda rows, **kwargs: rows
            facts = archive_qa._fact_hits("mainuser", "women", limit=4)
        finally:
            archive_qa.fact_bank.load_jsonl = original_load
            archive_qa.fact_bank.search = original_search
        self.assertEqual([fact["claim"] for fact in facts], ["all women are evil"])


class EmoteExplainPureTests(unittest.TestCase):
    def test_format_chat_keeps_emote_token_bare(self):
        report = {
            "name": "BatChest",
            "registry": {},
            "registry_tags": [],
            "has_vector": True,
            "usage_n": 30,
            "confidence": "strong",
            "neighbor_tags": [
                {"tag": "hype", "score": 1.2},
                {"tag": "meme", "score": 0.8},
            ],
            "neighbors": [
                {"name": "PogU", "score": 0.72, "tags": ["hype"]},
                {"name": "WOW", "score": 0.66, "tags": []},
            ],
            "axes": [
                {"name": "ironic", "score": 0.31, "label": "ironic"},
            ],
            "axis_note": None,
        }
        out = emote_explain.format_chat(report, detail=True, max_chars=220)
        self.assertTrue(out.startswith("BatChest "))
        self.assertNotIn("BatChest:", out)
        self.assertNotIn("basis", out)
        self.assertNotIn("confidence", out)
        self.assertIn("PogU", out)
        self.assertLessEqual(len(out), 220)

    def test_format_chat_raw_labels_sample_contexts(self):
        report = {
            "name": "BatChest",
            "registry": {},
            "has_vector": True,
            "usage_n": 30,
            "confidence": "strong",
            "neighbor_tags": [{"tag": "hype", "score": 1.2}],
            "neighbors": [{"name": "PogU", "score": 0.72}],
            "axes": [{"name": "ironic", "score": 0.31, "label": "ironic"}],
            "axis_note": None,
        }
        out = emote_explain.format_chat(report, raw=True, max_chars=180)
        self.assertIn("vector report", out)
        self.assertIn("vector_sample_contexts 30", out)
        self.assertIn("+0.72", out)
        self.assertNotIn("n=", out)
        self.assertLessEqual(len(out), 180)

    def test_clean_synthesis_keeps_emote_tokens_standalone(self):
        report = {
            "name": "KEKW",
            "query": "KEKW",
            "neighbors": [
                {"name": "docFaint"},
                {"name": "PEPW"},
                {"name": "ICANT"},
            ],
        }
        out = emote_explain.clean_synthesis(
            report,
            "KEKW <KEKW> is used to mean laughter. Similar emotes <docFaint PEPW ICANT>.",
        )
        self.assertTrue(out.startswith("KEKW is"))
        self.assertNotIn("<", out)
        self.assertNotIn("KEKW KEKW", out)
        self.assertNotIn("ICANT.", out)

    def test_clean_synthesis_does_not_prepend_before_quoted_emote(self):
        report = {"name": "tickpooJAWLINE", "query": "tickpooJAWLINE", "neighbors": []}
        out = emote_explain.clean_synthesis(report, "'tickpooJAWLINE' is used to mean x")
        self.assertEqual(out, "tickpooJAWLINE is used to mean x")

    def test_clean_synthesis_strips_llm_similar_tail(self):
        report = {
            "name": "CLARKSON",
            "query": "CLARKSON",
            "neighbors": [
                {"name": "QUIDES5"},
                {"name": "QUIDES5"},
                {"name": "SometimesMyGeniusIsAlmostFrightening"},
            ],
        }
        cleaned = emote_explain.clean_synthesis(
            report,
            "CLARKSON is used around Jeremy Clarkson bits. Similar emotes QUIDES5, QUIDES5, SometimesMyGeniusIsAlmostFrightening.",
        )
        self.assertEqual(cleaned, "CLARKSON is used around Jeremy Clarkson bits")
        out = emote_explain._append_similar_clause(report, cleaned, max_chars=200)
        self.assertIn("Similar emotes QUIDES5 SometimesMyGeniusIsAlmostFrightening", out)
        self.assertNotIn("QUIDES5,", out)
        self.assertNotIn("QUIDES5 QUIDES5", out)

    def test_archive_only_emote_uses_cautious_fallback(self):
        report = {
            "name": "CLARKSON",
            "query": "CLARKSON",
            "registry": {"origin": "ffz"},
            "registry_tags": [],
            "has_vector": False,
            "neighbor_tags": [],
            "neighbors": [],
            "archive": {
                "sampled": 90,
                "terms": [
                    {"term": "jeremy", "count": 13},
                    {"term": "generate", "count": 6},
                ],
            },
        }
        self.assertFalse(emote_explain._should_use_llm_synthesis(report))
        self.assertEqual(
            emote_explain.format_sentence(report),
            "CLARKSON appears in archive contexts around jeremy",
        )


class PersonaIqPureTests(unittest.TestCase):
    def test_roster_canonicalizes_aliases_and_drops_noise(self):
        roster = persona_iq._canonical_roster(["oldalt", "mainuser", "helperbot"])
        self.assertEqual(roster, ["mainuser"])


class ResidentPersonaPureTests(unittest.TestCase):
    def test_format_line_replaces_persona_label_with_prefix(self):
        state = {"persona": "normanbiz", "prefix": "tickpooJAWLINE \U0001f4e3"}
        line = resident_persona.format_line(state, "normanbiz: tickpooJAWLINE hello")
        self.assertEqual(line, "tickpooJAWLINE \U0001f4e3 hello")

    def test_format_line_does_not_return_prefix_only_message(self):
        state = {"persona": "normanbiz", "prefix": "tickpooJAWLINE \U0001f4e3"}
        self.assertEqual(resident_persona.format_line(state, "normanbiz:"), "")

    def test_normalize_state_clamps_probability_fields(self):
        state = resident_persona._normalize_state(
            "ThickPoo",
            {
                "persona": "NormanBiz",
                "chance": 2,
                "topic_chance": 3,
                "topic_curve": -1,
                "directed_chance": -1,
                "directed_cooldown": -3,
                "idle_chance": -2,
                "cooldown": -5,
                "reply_to_trigger": "off",
            },
        )
        self.assertEqual(state["channel"], "thickpoo")
        self.assertEqual(state["persona"], "normanbiz")
        self.assertEqual(state["chance"], 1.0)
        self.assertEqual(state["topic_chance"], 1.0)
        self.assertEqual(state["topic_curve"], 0.25)
        self.assertEqual(state["directed_chance"], 0.0)
        self.assertEqual(state["directed_cooldown"], 0.0)
        self.assertEqual(state["idle_chance"], 0.0)
        self.assertEqual(state["cooldown"], 0.0)
        self.assertFalse(state["reply_to_trigger"])


class PersonaRetrievalPureTests(unittest.TestCase):
    def test_repeated_token_spam_is_not_usable_persona_evidence(self):
        self.assertFalse(persona_llm._usable_exemplar("FART Fart FART Fart FART Fart"))
        self.assertFalse(persona_llm._usable_exemplar("@Quin69 FART AYAP Fart " * 6))
        self.assertTrue(persona_llm._usable_exemplar("post your fart in chat rn"))

    def test_directed_prompt_is_weighted_over_recent_context(self):
        recent = [
            ("2026-01-01 00:00:01", "a", "thats such an answer"),
            ("2026-01-01 00:00:02", "b", "and he got it wrong too"),
        ]
        text = persona_llm._retrieval_text(recent, "favorite game")
        self.assertEqual(text.count("favorite game"), 3)

    def test_conversation_rows_dedupes_alias_repeated_lines(self):
        recent = [
            ("2026-01-01 00:00:01", "oldalt", "same copied line"),
            ("2026-01-01 00:00:02", "mainuser", "same copied line"),
            ("2026-01-01 00:00:03", "otheruser", "~persona mainuser hi"),
            ("2026-01-01 00:00:04", "otheruser", "fresh line"),
        ]
        rows = persona_llm._conversation_rows(recent)
        self.assertEqual([(author, content) for _ts, author, content in rows],
                         [("oldalt", "same copied line"), ("otheruser", "fresh line")])

    def test_early_direct_terms_outrank_later_context_terms(self):
        ranked = persona_llm._rank_relevant_texts(
            ["well see how this goes", "fart is real in this game"],
            [],
            ["fart", "bragging", "well", "see"],
        )
        self.assertEqual(ranked[0], "fart is real in this game")

    def test_semantic_hit_needs_higher_score_without_query_anchor(self):
        terms = ["fart"]
        self.assertTrue(persona_llm._semantic_text_allowed("fart is real", 0.51, terms))
        self.assertFalse(persona_llm._semantic_text_allowed("bragging about winning", 0.59, terms))
        self.assertTrue(persona_llm._semantic_text_allowed("bragging about winning", 0.63, terms))

    def test_keyword_evidence_not_blindly_displaced_by_semantic_hit(self):
        ranked = persona_llm._rank_relevant_texts(
            ["i keep farting in this game"],
            [(0.63, "bragging about winning again")],
            ["fart"],
        )
        self.assertEqual(ranked[0], "i keep farting in this game")

    def test_heldout_eval_can_remove_target_from_prompt_evidence(self):
        signature, relevant, snippets = persona_llm._filter_excluded_evidence(
            ["normal style line", "the hidden target reply"],
            ["another relevant line", "The hidden target reply!!!"],
            [">> mainuser: the hidden target reply\nother: ok", "other: harmless"],
            ["hidden target reply"],
        )
        self.assertEqual(signature, ["normal style line"])
        self.assertEqual(relevant, ["another relevant line"])
        self.assertEqual(snippets, ["other: harmless"])


if __name__ == "__main__":
    unittest.main()
