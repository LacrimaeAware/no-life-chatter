import asyncio
import importlib
import math
import sys
import threading
import types
import unittest
import urllib.error
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
        DB_PATH="data/unsynced/test_settings.db",
        LLM_ENDPOINT="http://127.0.0.1:1234/v1/chat/completions",
        LLM_MODEL="local",
        LLM_EMBED_MODEL="text-embedding-bge-m3",
        LLM_MODEL_SHORTCUTS={},
        COMEDY_CHANNELS=(),
        CHANNELS=("mainroom",),
        LLM_SEMANTIC_MIN_SCORE=0.50,
        LLM_SEMANTIC_UNANCHORED_MIN_SCORE=0.62,
    )
    sys.modules["config"] = fake
    return fake


install_fake_config()
sys.modules["services.llm"] = types.SimpleNamespace(chat=None)
from utils import archive_qa, chat_archive, emote_explain, fact_bank, message_quality, persona_classifier, persona_iq, persona_llm, resident_persona, user_profiles  # noqa: E402
from utils import persona_axes  # noqa: E402
from services import model_queue  # noqa: E402
from commands import markers  # noqa: E402
from command_processor import _backend_offline, _backend_rejected, _model_command_kind  # noqa: E402


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

    def test_pick_display_prefers_most_recently_seen_raw_name(self):
        keys = ["altone", "mainuser", "oldalt"]
        # The canonical's message recency is poisoned (live rows are stored
        # under it), so only its checked_at counts; the alt seen live later
        # wins even though the canonical has newer message rows.
        checked = {"mainuser": "2026-06-12 17:09:21", "altone": "2026-07-09 19:30:15"}
        sent = {"mainuser": "2026-07-09 19:45:00", "oldalt": "2025-05-12 02:04:10"}
        self.assertEqual(
            chat_archive._pick_display("mainuser", keys, checked, sent), "altone")

    def test_pick_display_keeps_canonical_when_it_is_the_active_account(self):
        keys = ["altone", "mainuser", "oldalt"]
        checked = {"mainuser": "2026-07-09 19:42:35", "altone": "2026-07-01 09:09:45"}
        sent = {"oldalt": "2020-01-24 22:53:40"}
        self.assertEqual(
            chat_archive._pick_display("mainuser", keys, checked, sent), "mainuser")

    def test_pick_display_falls_back_to_messages_when_never_seen_live(self):
        keys = ["altone", "mainuser", "oldalt"]
        sent = {"oldalt": "2024-01-01 00:00:00", "mainuser": "2023-01-01 00:00:00"}
        self.assertEqual(
            chat_archive._pick_display("mainuser", keys, {}, sent), "oldalt")

    def test_display_name_short_circuits_for_unaliased_user(self):
        self.assertEqual(chat_archive.display_name("solochatter"), "solochatter")

    def test_pick_casing_prefers_how_chat_types_the_name(self):
        forms = ["MainUserGuy", "MainUserGuy", "mainuserguy", "MAINUSERGUY"]
        self.assertEqual(chat_archive._pick_casing(forms, "mainuserguy"),
                         "MainUserGuy")

    def test_pick_casing_needs_two_sightings_and_exact_casefold(self):
        self.assertEqual(chat_archive._pick_casing(["OneOff"], "oneoff"), "oneoff")
        self.assertEqual(chat_archive._pick_casing([], "nobody"), "nobody")

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


class CommandProcessorPureTests(unittest.TestCase):
    def test_http_400_model_error_is_not_offline(self):
        exc = urllib.error.HTTPError(
            "http://127.0.0.1:1234/v1/embeddings",
            400,
            "Bad Request",
            None,
            None,
        )
        self.assertFalse(_backend_offline(exc))
        self.assertTrue(_backend_rejected(exc))

    def test_url_timeout_is_busy_not_offline(self):
        exc = urllib.error.URLError(TimeoutError("timed out"))
        self.assertFalse(_backend_offline(exc))
        self.assertTrue(_backend_rejected(exc))

    def test_model_command_kind_routes_heavy_and_fast_paths(self):
        self.assertEqual(_model_command_kind("persona", ["mainuser"]), "required")
        self.assertEqual(_model_command_kind("distinct", ["top"]), "required")
        self.assertEqual(_model_command_kind("askchat", ["mainuser", "math"]), "optional")
        self.assertIsNone(_model_command_kind("askchat", ["raw", "mainuser", "math"]))
        self.assertIsNone(_model_command_kind("emote", ["BatChest", "raw"]))
        self.assertIsNone(_model_command_kind("generate", ["list"]))
        self.assertIsNone(_model_command_kind("generate", ["mainuser", "engine=markov"]))
        original = persona_axes.axis_cached
        try:
            persona_axes.axis_cached = lambda term: False
            self.assertEqual(_model_command_kind("top", ["uncachedtrait"]), "required")
            persona_axes.axis_cached = lambda term: term == "known"
            self.assertIsNone(_model_command_kind("top", ["known"]))
        finally:
            persona_axes.axis_cached = original

    def test_why_only_queues_for_uncached_axis_or_words_mode(self):
        original = persona_axes.axis_cached
        try:
            persona_axes.axis_cached = lambda term: term == "known"
            self.assertIsNone(_model_command_kind("why", ["mainuser", "known"]))
            self.assertEqual(_model_command_kind("why", ["mainuser", "newtrait"]), "required")
            self.assertEqual(_model_command_kind("why", ["mainuser", "known", "words"]), "required")
        finally:
            persona_axes.axis_cached = original


class ModelQueuePureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_state = model_queue.server_state

        async def up():
            return "up"

        model_queue.server_state = up
        model_queue.clear_pending()

    async def asyncTearDown(self):
        model_queue.clear_pending()
        model_queue.server_state = self.original_state

    async def test_queue_finishes_one_handler_before_starting_next(self):
        events = []

        async def send(text):
            events.append(("send", text))

        async def first():
            events.append(("work", "first-start"))
            await asyncio.sleep(0.02)
            events.append(("work", "first-done"))

        async def second():
            events.append(("work", "second-start"))
            events.append(("work", "second-done"))

        task1 = asyncio.create_task(model_queue.submit(
            label="first", work=first, send=send, user="a", user_key="a", sig="first"))
        await asyncio.sleep(0.005)
        task2 = asyncio.create_task(model_queue.submit(
            label="second", work=second, send=send, user="b", user_key="b", sig="second"))
        await asyncio.gather(task1, task2)

        self.assertLess(events.index(("work", "first-done")),
                        events.index(("work", "second-start")))
        self.assertLess(events.index(("send", "@b Processing...")),
                        events.index(("work", "second-start")))


class PersonaAxisPureTests(unittest.TestCase):
    def test_axis_error_message_explains_embedding_busy(self):
        persona_axes._set_axis_error(
            "breedable",
            'embedding HTTP 400: {"error":"Model has not started loading/has been unloaded."}',
        )
        self.assertIn("busy/loading", persona_axes.axis_error_message("breedable"))


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
        self.assertIn("No clear archive receipts", out)
        self.assertNotIn("Weak one-off", out)
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
                {"name": "QUIRKY5"},
                {"name": "QUIRKY5"},
                {"name": "SometimesMyGeniusIsAlmostFrightening"},
            ],
        }
        cleaned = emote_explain.clean_synthesis(
            report,
            "CLARKSON is used around Jeremy Clarkson bits. Similar emotes QUIRKY5, QUIRKY5, SometimesMyGeniusIsAlmostFrightening.",
        )
        self.assertEqual(cleaned, "CLARKSON is used around Jeremy Clarkson bits")
        out = emote_explain._append_similar_clause(report, cleaned, max_chars=200)
        self.assertIn("Similar emotes QUIRKY5 SometimesMyGeniusIsAlmostFrightening", out)
        self.assertNotIn("QUIRKY5,", out)
        self.assertNotIn("QUIRKY5 QUIRKY5", out)

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

    def test_rarity_excludes_emote_names_and_usernames(self):
        freqs = Counter({"epistemology": 5, "omegalul": 500, "mainuser": 40})
        rarity = persona_iq._rarity_fn(freqs, 20000,
                                       exclusions={"omegalul", "mainuser"})
        self.assertIsNone(rarity("omegalul"))
        self.assertIsNone(rarity("OMEGALUL"))
        self.assertIsNone(rarity("mainuser"))
        self.assertIsNotNone(rarity("epistemology"))


class UserProfilesPureTests(unittest.TestCase):
    def _judged(self, value, sent_at, sincerity="sincere", asserts=True):
        # distinct phrasing per row: verbatim repeats are deduped by design
        # (a running gag repeated word-for-word must not corroborate itself)
        return {"asserts": asserts, "value": value, "sincerity": sincerity,
                "id": 1, "sent_at": sent_at, "channel": "mainroom",
                "content": f"i said {value} at {sent_at}"}

    def test_parse_judgment_tolerates_prose_around_json(self):
        out = user_profiles._parse_judgment(
            'Sure! Here is the answer:\n{"asserts": true, "value": "Poland", '
            '"sincerity": "sincere"} hope that helps')
        self.assertEqual(out, {"asserts": True, "value": "poland",
                               "sincerity": "sincere"})
        self.assertIsNone(user_profiles._parse_judgment("no json here"))
        self.assertIsNone(user_profiles._parse_judgment(None))

    def test_reconcile_confirms_only_with_multi_day_support(self):
        one_day = [self._judged("poland", "2026-01-01 10:00:00"),
                   self._judged("poland", "2026-01-01 11:00:00")]
        self.assertEqual(user_profiles._reconcile("location", one_day)["status"],
                         "candidate")
        two_days = one_day + [self._judged("poland", "2026-02-01 10:00:00")]
        self.assertEqual(user_profiles._reconcile("location", two_days)["status"],
                         "confirmed")

    def test_placeholder_and_anecdote_values_rejected(self):
        judged = [self._judged("my country", "2026-01-01 10:00:00"),
                  self._judged("my country", "2026-01-02 10:00:00")]
        self.assertIsNone(user_profiles._reconcile("location", judged))
        self.assertFalse(user_profiles._valid_value("i lift way more than 5 lbs"))
        self.assertFalse(user_profiles._valid_value("still working on july 2nd at work"))
        self.assertTrue(user_profiles._valid_value("germany"))
        self.assertTrue(user_profiles._valid_value("software developer"))

    def test_verbatim_repeats_never_corroborate(self):
        gag = {"asserts": True, "value": "husband", "sincerity": "sincere",
               "id": 1, "channel": "mainroom",
               "content": "i was devastated when i found out my wife was cheating on me"}
        judged = [dict(gag, sent_at="2022-09-17 10:00:00"),
                  dict(gag, sent_at="2022-09-27 10:00:00")]
        out = user_profiles._reconcile("relationship", judged)
        self.assertEqual(out["status"], "candidate")

    def test_reconcile_marks_conflicting_confirmed_values_disputed(self):
        judged = [self._judged("poland", "2026-01-01 10:00:00"),
                  self._judged("poland", "2026-01-02 10:00:00"),
                  self._judged("brazil", "2026-03-01 10:00:00"),
                  self._judged("brazil", "2026-03-02 10:00:00")]
        out = user_profiles._reconcile("location", judged)
        self.assertEqual(out["status"], "disputed")
        self.assertTrue(out["alternatives"])

    def test_reconcile_drops_jokes_and_nonassertions(self):
        judged = [self._judged("a panda", "2026-01-01 10:00:00", sincerity="joke"),
                  self._judged("a panda", "2026-01-02 10:00:00", asserts=False)]
        self.assertIsNone(user_profiles._reconcile("pets", judged))

    def test_adjacent_ages_do_not_dispute(self):
        judged = [self._judged("25", "2025-01-01 10:00:00"),
                  self._judged("25", "2025-06-01 10:00:00"),
                  self._judged("26", "2026-01-01 10:00:00"),
                  self._judged("26", "2026-01-02 10:00:00")]
        out = user_profiles._reconcile("age", judged)
        self.assertEqual(out["status"], "confirmed")
        self.assertNotIn("alternatives", out)

    def test_multi_slot_keeps_independent_values(self):
        judged = [self._judged("dog", "2026-01-01 10:00:00"),
                  self._judged("dog", "2026-01-05 10:00:00"),
                  self._judged("cat", "2026-02-01 10:00:00")]
        out = user_profiles._reconcile("pets", judged)
        statuses = {v["value"]: v["status"] for v in out["values"]}
        self.assertEqual(statuses, {"dog": "confirmed", "cat": "candidate"})


class PersonaOutputPureTests(unittest.TestCase):
    def test_clean_output_skips_meta_preamble_for_target_labeled_line(self):
        raw = (
            "Based on the chat history and the style of Bluepigman5000, "
            "their next message could be:\n"
            "Bluepigman5000: DuardoCry"
        )
        self.assertEqual(persona_llm._clean_output(raw, "Bluepigman5000"), "DuardoCry")

    def test_candidate_rejects_assistant_preamble(self):
        text = "Based on the chat history and the style of Bluepigman5000, their next message could be:"
        self.assertEqual(
            persona_llm._candidate_issues("Bluepigman5000", text, []),
            "assistant preamble",
        )

    def test_candidate_rejects_nested_speaker_label(self):
        cleaned = persona_llm._clean_output("Bluepigman5000: duardo1: peepoZ", "Bluepigman5000")
        self.assertEqual(cleaned, "duardo1: peepoZ")
        self.assertEqual(
            persona_llm._candidate_issues("Bluepigman5000", cleaned, []),
            "speaker label",
        )


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
            "TickPoo",
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
        self.assertEqual(state["channel"], "tickpoo")
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
