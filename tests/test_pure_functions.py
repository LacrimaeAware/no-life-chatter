import asyncio
import importlib
import json
import math
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
from collections import Counter
from pathlib import Path


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
from utils import archive_qa, artifact_status, atomic_file, chat_archive, emote_explain, fact_bank, irony, message_quality, persona_classifier, persona_embeddings, persona_iq, persona_llm, persona_msg_index, resident_persona, user_profiles  # noqa: E402
from utils import persona_axes  # noqa: E402
from services import model_queue  # noqa: E402
from commands import markers, why  # noqa: E402
from command_processor import _backend_offline, _backend_rejected, _model_command_kind  # noqa: E402


def tearDownModule():
    chat_archive.close_thread_connection()


class AtomicFileTests(unittest.TestCase):
    def test_atomic_write_replaces_only_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.txt"
            path.write_text("old", encoding="utf-8")
            with atomic_file.open_atomic(path, "w", encoding="utf-8") as handle:
                handle.write("new")
            self.assertEqual(path.read_text(encoding="utf-8"), "new")
            with self.assertRaises(RuntimeError):
                with atomic_file.open_atomic(path, "w", encoding="utf-8") as handle:
                    handle.write("partial")
                    raise RuntimeError("stop")
            self.assertEqual(path.read_text(encoding="utf-8"), "new")


class AxisParsingTests(unittest.TestCase):
    def test_trailing_comma_repair_preserves_json_closers(self):
        raw = '{"items":["one", "two",], "ok":true,}'
        cleaned = persona_axes._clean_json_trailing_commas(raw)
        self.assertEqual(cleaned, '{"items":["one", "two"], "ok":true}')
        self.assertNotIn("\x01", cleaned)


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

    def test_utterance_merge_never_crosses_channel_boundaries(self):
        rows = [
            ("2026-07-15 12:00:00", "room_a", "user", "first"),
            ("2026-07-15 12:00:05", "room_b", "user", "unrelated"),
            ("2026-07-15 12:00:10", "room_a", "user", "continued"),
        ]
        merged = chat_archive.merge_channel_utterances(rows, gap_seconds=45)
        self.assertEqual(
            merged,
            [
                ("2026-07-15 12:00:00", "room_a", "user", "first continued"),
                ("2026-07-15 12:00:05", "room_b", "user", "unrelated"),
            ],
        )

    def test_utterance_records_keep_component_ids(self):
        rows = [
            (10, "2026-07-15 12:00:00", "room_a", "user", "first"),
            (11, "2026-07-15 12:00:05", "room_b", "user", "other"),
            (12, "2026-07-15 12:00:10", "room_a", "user", "continued"),
        ]
        merged = chat_archive.merge_channel_utterance_records(rows, gap_seconds=45)
        self.assertEqual(merged[0]["text"], "first continued")
        self.assertEqual(merged[0]["message_ids"], [10, 12])
        self.assertEqual(merged[0]["parts"], [(10, "first"), (12, "continued")])
        self.assertEqual(merged[1]["text"], "other")

    def test_utterance_merge_collapses_duplicate_components(self):
        rows = [
            (10, "2026-07-15 12:00:00", "room_a", "user", "same line"),
            (11, "2026-07-15 12:00:05", "room_a", "user", "Same line!"),
            (12, "2026-07-15 12:00:10", "room_a", "user", "new thought"),
        ]
        merged = chat_archive.merge_channel_utterance_records(rows, gap_seconds=45)
        self.assertEqual(merged[0]["text"], "same line new thought")
        self.assertEqual(merged[0]["message_ids"], [10, 11, 12])
        self.assertEqual(len(merged[0]["parts"]), 3)

    def test_author_alias_chains_and_author_keys(self):
        self.assertEqual(chat_archive.normalize_author("@OldAlt,"), "mainuser")
        self.assertEqual(chat_archive.author_keys("oldalt"), ["altone", "mainuser", "oldalt"])

    def test_canonical_roster_aggregates_aliases_before_cutoff(self):
        class FakeConnection:
            def execute(self, _query):
                return [
                    ("oldalt", 4),
                    ("mainuser", 3),
                    ("someone", 6),
                    ("helperbot", 99),
                ]

        original_connect = chat_archive.connect
        try:
            chat_archive.connect = lambda: FakeConnection()
            self.assertEqual(
                chat_archive.canonical_author_counts(limit=2),
                [("mainuser", 7), ("someone", 6)],
            )
        finally:
            chat_archive.connect = original_connect

    def test_archive_connections_are_thread_local(self):
        main_conn = chat_archive.connect()
        seen = []

        def worker():
            seen.append(chat_archive.connect())
            chat_archive.close_thread_connection()

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
        from utils import persona_traits

        original_ready = persona_traits.axes_ready
        persona_traits.axes_ready = lambda: False
        self.addCleanup(setattr, persona_traits, "axes_ready", original_ready)
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
            persona_traits.axes_ready = lambda: True
            self.assertIsNone(_model_command_kind("distinct", []))
            self.assertIsNone(_model_command_kind("traits", ["mainuser"]))
        finally:
            persona_axes.axis_cached = original

    def test_why_only_queues_for_uncached_axis_or_words_mode(self):
        from utils import persona_traits

        original = persona_axes.axis_cached
        original_ready = persona_traits.axes_ready
        try:
            persona_traits.axes_ready = lambda: True
            persona_axes.axis_cached = lambda term: term == "known"
            self.assertIsNone(_model_command_kind("why", ["mainuser", "known"]))
            self.assertEqual(_model_command_kind("why", ["mainuser", "newtrait"]), "required")
            self.assertEqual(_model_command_kind("why", ["mainuser", "known", "words"]), "required")
        finally:
            persona_axes.axis_cached = original
            persona_traits.axes_ready = original_ready

    def test_cold_builtin_axis_queues_once(self):
        from utils import persona_traits

        original_ready = persona_traits.axes_ready
        try:
            persona_traits.axes_ready = lambda: False
            self.assertEqual(_model_command_kind("top", ["professor"]), "required")
            self.assertEqual(
                _model_command_kind("why", ["mainuser", "professor"]),
                "required",
            )
            persona_traits.axes_ready = lambda: True
            self.assertIsNone(_model_command_kind("top", ["professor"]))
            self.assertIsNone(_model_command_kind("why", ["mainuser", "professor"]))
        finally:
            persona_traits.axes_ready = original_ready


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


class WhyCommandPureTests(unittest.TestCase):
    def test_squash_repeated_text_halves(self):
        text = "i am intelligent i am intelligent"
        self.assertEqual(why._squash_repeated_text(text), "i am intelligent")

    def test_axis_basis_note_exposes_opposite_custom_axis(self):
        axes = {"goy": (object(), "goy", "sophisticated")}
        self.assertEqual(
            why._axis_basis_note("sophisticated", "goy", -1, axes["goy"][1], axes["goy"][2]),
            "basis sophisticated vs goy (axis goy)",
        )


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
    def test_training_messages_are_normalized_exact_deduped(self):
        messages = [
            "This is my repeated line!",
            "this is my repeated line",
            "A genuinely different line here",
        ]
        self.assertEqual(
            persona_classifier._unique_usable_messages(messages),
            ["This is my repeated line!", "A genuinely different line here"],
        )

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

    def test_detects_model_requests_and_generated_response_candidates(self):
        self.assertTrue(message_quality.model_request_like("<gemini3 is this true"))
        self.assertFalse(message_quality.model_request_like("what does gemini mean"))
        response = (
            "There is no publicly available information confirming that claim. "
            "The available record is inconclusive."
        )
        self.assertTrue(message_quality.generated_response_candidate(response))

    def test_pasted_prose_filter_is_conservative(self):
        pasted = (
            "Zarathustra contrasts the ideal with the last man of modernity, "
            "an alternative goal which humanity might set for itself. The last "
            "man appears only in the later work and is presented as a smothering "
            "of aspiration. This paragraph continues in formal reference prose."
        )
        self.assertTrue(message_quality.likely_pasted_prose(pasted))
        self.assertFalse(message_quality.likely_pasted_prose(
            "i think the sample is biased because only regulars answered"
        ))

    def test_keeps_reasonable_semantic_text(self):
        text = "because jupyter needs the kernel packages installed for interactive python"
        self.assertTrue(message_quality.usable_for_iq(text))
        self.assertIsNotNone(message_quality.semantic_text(text))

    def test_semantic_selection_is_deterministic_and_deduped(self):
        messages = [
            f"this is ordinary discussion number word {i} about a different topic"
            for i in range(30)
        ]
        messages += [messages[0], messages[0].upper()]
        first, meta = message_quality.select_semantic_units(
            messages, cap=10, label="sample-user"
        )
        second, _ = message_quality.select_semantic_units(
            messages, cap=10, label="sample-user"
        )
        self.assertEqual([row["key"] for row in first], [row["key"] for row in second])
        self.assertEqual(len({row["key"] for row in first}), 10)
        self.assertEqual(meta["coverage"], 8)
        self.assertEqual(meta["high_signal"], 2)


class FactBankPureTests(unittest.TestCase):
    def test_fact_bank_metadata_tracks_identity_provenance(self):
        meta = {
            "version": fact_bank.VERSION,
            "alias_signature": chat_archive.alias_signature(),
            "content_sha256": "abc",
        }
        self.assertTrue(fact_bank.metadata_current(meta))
        meta["alias_signature"] = "old"
        self.assertFalse(fact_bank.metadata_current(meta))

    def test_fact_bank_hash_rejects_mismatched_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facts.jsonl"
            fact_bank.write_jsonl([{"author": "mainuser", "claim": "one"}], path)
            self.assertEqual(len(fact_bank.load_jsonl(path)), 1)
            path.write_text('{"author":"mainuser","claim":"tampered"}\n', encoding="utf-8")
            self.assertEqual(fact_bank.load_jsonl(path), [])

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

    def test_rejects_clause_shaped_possession_value(self):
        rows = fact_bank.extract_claims(
            "mainuser",
            "I hate the arguments of my country is better than your country when people lose",
        )
        self.assertFalse(any(row["kind"] == "possession" for row in rows))

    def test_possession_value_stops_at_discourse_connector(self):
        rows = fact_bank.extract_claims(
            "mainuser",
            "yes but my degree is in math and statistics so i need another focus",
        )
        self.assertIn(
            ("possession", "degree = in math and statistics"),
            {(row["kind"], row["claim"]) for row in rows},
        )

    def test_claim_tail_drops_trailing_mention(self):
        rows = fact_bank.extract_claims("mainuser", "I hate math @OtherUser")
        self.assertIn(
            ("preference_negative", "math"),
            {(row["kind"], row["claim"]) for row in rows},
        )


class ArchiveQaPureTests(unittest.TestCase):
    def test_dense_metadata_recovery_reconstructs_merged_utterance(self):
        class EmptyCursor:
            def fetchall(self):
                return []

        class FakeConnection:
            def execute(self, _query, _params):
                return EmptyCursor()

        original_connect = archive_qa.chat_archive.connect
        original_records = archive_qa.chat_archive.utterance_records_for
        try:
            archive_qa.chat_archive.connect = lambda: FakeConnection()
            archive_qa.chat_archive.utterance_records_for = lambda *_args, **_kwargs: [{
                "id": 10,
                "sent_at": "2026-07-15 12:00:00",
                "channel": "mainroom",
                "author": "mainuser",
                "text": "first continued",
                "message_ids": [10, 12],
            }]
            recovered = archive_qa._recover_author_meta(
                "mainuser", ["first continued"]
            )
        finally:
            archive_qa.chat_archive.connect = original_connect
            archive_qa.chat_archive.utterance_records_for = original_records
        row = recovered[chat_archive.line_match_key("first continued")]
        self.assertEqual(row["message_ids"], [10, 12])

    def test_disputed_profile_keeps_supported_alternative(self):
        data = {
            "value": "poland",
            "status": "disputed",
            "alternatives": [
                {"value": "canada", "status": "confirmed"},
                {"value": "france", "status": "candidate"},
            ],
        }
        self.assertEqual(
            [entry["value"] for entry in archive_qa._profile_entries(data)],
            ["poland", "canada"],
        )

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

    def test_attach_context_uses_human_rows_and_marks_receipt(self):
        original = chat_archive.context_window
        try:
            chat_archive.context_window = lambda *args, **kwargs: [
                (9, "other", "did you actually finish it"),
                (10, "oldalt", "yes because the parser was broken"),
                (11, "helperbot", "~automated response"),
            ]
            hits = archive_qa._attach_context([{
                "id": 10,
                "author": "mainuser",
                "channel": "mainroom",
                "sent_at": "2026-01-01 00:00:00",
                "text": "yes because the parser was broken",
            }])
        finally:
            chat_archive.context_window = original
        self.assertEqual(len(hits[0]["context"]), 2)
        self.assertTrue(hits[0]["context"][1]["hit"])
        self.assertEqual(hits[0]["context"][1]["author"], "mainuser")

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
                "support_days": 2,
                "unique_phrasings": 2,
                "evidence_confidence": 0.6,
                "status": "corroborated_claim",
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
        self.assertNotIn("terms=", out)
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

    def test_evidence_items_keep_high_confidence_dense_paraphrase(self):
        report = {
            "query": "does he like cars",
            "author": "mainuser",
            "channel": None,
            "terms": ["cars"],
            "facts": [],
            "archive": [{
                "author": "mainuser",
                "channel": "mainroom",
                "sent_at": "2026-01-01 00:00:00",
                "text": "my civic absolutely rips",
                "lanes": ["dense"],
                "dense_score": 0.72,
            }],
            "near": [],
            "emotes": [],
        }
        evidence = archive_qa.evidence_items(report)
        self.assertIn("my civic absolutely rips", evidence[0]["text"])

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

    def test_unscoped_fact_hits_require_the_claim_owner_in_the_question(self):
        rows = [
            {
                "author": "mainuser",
                "kind": "self_identity",
                "claim": "dead",
                "support_count": 2,
                "confidence": 0.8,
                "evidence": [{"clean_text": "i am dead"}],
            },
            {
                "author": "someoneelse",
                "kind": "self_identity",
                "claim": "dead",
                "support_count": 2,
                "confidence": 0.8,
                "evidence": [{"clean_text": "i am dead too"}],
            },
        ]
        original_load = archive_qa.fact_bank.load_jsonl
        original_search = archive_qa.fact_bank.search
        try:
            archive_qa.fact_bank.load_jsonl = lambda: rows
            archive_qa.fact_bank.search = lambda rows, **kwargs: rows
            facts = archive_qa._fact_hits(None, "is mainuser dead?", limit=4)
        finally:
            archive_qa.fact_bank.load_jsonl = original_load
            archive_qa.fact_bank.search = original_search
        self.assertEqual([fact["author"] for fact in facts], ["mainuser"])


class EmoteExplainPureTests(unittest.TestCase):
    def test_emote_semantic_checkpoint_path_is_not_live_artifact(self):
        from scripts import build_emote_semantics

        self.assertEqual(
            build_emote_semantics._partial_path("folder/emotes.pkl"),
            "folder/emotes.partial.pkl",
        )

    def test_emote_checkpoint_signature_normalizes_requested_case_and_order(self):
        from scripts import build_emote_semantics

        first = build_emote_semantics._request_signature(
            top=20, contexts=160, emotes="KEKW,BatChest", refresh=True
        )
        second = build_emote_semantics._request_signature(
            top=20, contexts=160, emotes="batchest,kekw", refresh=True
        )
        self.assertEqual(first, second)

    def test_emote_semantics_reader_accepts_versioned_wrapper(self):
        import pickle

        from utils import emote_meaning

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emote_semantics.pkl"
            with path.open("wb") as handle:
                pickle.dump({
                    "__meta__": {"version": 2},
                    "emotes": {"ExampleEmote": {"vector": [1.0, 0.0], "n": 160}},
                }, handle)
            original_path = emote_meaning.SEM_PATH
            original_state = (
                emote_meaning._sem,
                emote_meaning._centered,
                emote_meaning._names,
                emote_meaning._sem_stamp,
            )
            try:
                emote_meaning.SEM_PATH = str(path)
                emote_meaning._sem = None
                emote_meaning._sem_stamp = None
                self.assertEqual(emote_meaning.usage_count("exampleemote"), 160)
                self.assertEqual(emote_meaning.semantics_metadata()["version"], 2)
            finally:
                emote_meaning.SEM_PATH = original_path
                (
                    emote_meaning._sem,
                    emote_meaning._centered,
                    emote_meaning._names,
                    emote_meaning._sem_stamp,
                ) = original_state

    def test_emote_context_builder_uses_previous_line_for_bare_reaction(self):
        from scripts import build_emote_semantics

        text, used_previous = build_emote_semantics._clean_context(
            "CLARKSON", "the new episode starts tomorrow", "CLARKSON"
        )
        self.assertTrue(used_previous)
        self.assertIn("episode", text)

    def test_emote_context_builder_strips_token_case_insensitively(self):
        from scripts import build_emote_semantics

        text, used_previous = build_emote_semantics._clean_context(
            "batChest this launch is wildly overhyped", "", "BatChest"
        )
        self.assertFalse(used_previous)
        self.assertNotIn("batchest", text.casefold())

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
    def test_required_iq_quality_rejects_missing_model_phases(self):
        failures = persona_iq._quality_failures(
            embedding_requested=True,
            embedding_authors=0,
            judge_requested=True,
            judged_authors=5,
            total_authors=40,
        )
        self.assertEqual(len(failures), 2)
        self.assertIn("embeddings covered 0/40", failures[0])
        self.assertIn("judge covered 5/40", failures[1])

    def test_optional_iq_phases_do_not_degrade_lexical_build(self):
        self.assertEqual(persona_iq._quality_failures(
            embedding_requested=False,
            embedding_authors=0,
            judge_requested=False,
            judged_authors=0,
            total_authors=40,
        ), [])

    def test_utterance_rows_dedupes_repeated_personal_lines(self):
        original = persona_iq.chat_archive.utterance_records_for
        try:
            persona_iq.chat_archive.utterance_records_for = lambda _author: [
                {"id": 1, "text": "because this model predicts the same result",
                 "parts": [(1, "because this model predicts the same result")]},
                {"id": 2, "text": "Because this model predicts the same result!",
                 "parts": [(2, "Because this model predicts the same result!")]},
                {"id": 3, "text": "however the counterexample changes the conclusion",
                 "parts": [(3, "however the counterexample changes the conclusion")]},
            ]
            rows = persona_iq._utterance_rows("mainuser", author_cap=100)
        finally:
            persona_iq.chat_archive.utterance_records_for = original
        self.assertEqual(len(rows), 2)

    def test_utterance_rows_rejects_command_generated_reply(self):
        original_records = persona_iq.chat_archive.utterance_records_for
        original_previous = persona_iq._preceded_by_model_request
        try:
            generated = (
                "There is no publicly available information confirming that claim. "
                "The evidence remains inconclusive."
            )
            persona_iq.chat_archive.utterance_records_for = lambda _author: [
                {"id": 1, "text": generated, "parts": [(1, generated)]},
                {"id": 2,
                 "text": "because the sample changes the conclusion <gemini3 explain it",
                 "parts": [
                     (2, "because the sample changes the conclusion"),
                     (3, "<gemini3 explain it"),
                 ]},
            ]
            persona_iq._preceded_by_model_request = lambda message_id: message_id == 1
            rows = persona_iq._utterance_rows("mainuser", author_cap=100)
        finally:
            persona_iq.chat_archive.utterance_records_for = original_records
            persona_iq._preceded_by_model_request = original_previous
        self.assertEqual([row["raw"] for row in rows], [
            "because the sample changes the conclusion"
        ])

    def test_iq_cache_requires_current_identity_and_utterance_provenance(self):
        payload = {"__meta__": {
            "version": persona_iq.VERSION,
            "alias_signature": chat_archive.alias_signature(),
            "utterance_version": chat_archive.UTTERANCE_VERSION,
        }}
        self.assertTrue(persona_iq._cache_current(payload))
        payload["__meta__"]["utterance_version"] -= 1
        self.assertFalse(persona_iq._cache_current(payload))

    def test_iq_cache_rejects_degraded_model_coverage(self):
        payload = {"__meta__": {
            "version": persona_iq.VERSION,
            "alias_signature": chat_archive.alias_signature(),
            "utterance_version": chat_archive.UTTERANCE_VERSION,
            "build_quality": "degraded",
            "quality_failures": ["judge covered 0/40 authors"],
        }}
        self.assertFalse(persona_iq._cache_current(payload))

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

    def test_tail_receipts_show_top_tail_median_before_maximum(self):
        rows = persona_iq._tail_receipts(
            [(value, f"reasoning example number {value} with enough text")
             for value in range(20)],
            "causal",
        )
        self.assertEqual([row["value"] for row in rows], [18.0, 19.0])

    def test_cross_author_long_copypasta_is_removed(self):
        copied = "this is a sufficiently long repeated definition copied into chat"
        unique = "this is a sufficiently long original explanation from one person"
        make = lambda text: {"raw": text, "clean": text, "tokens": text.split()}
        filtered, removed = persona_iq._drop_cross_author_copies({
            "a": [make(copied), make(unique)],
            "b": [make(copied)],
        })
        self.assertEqual(removed, 2)
        self.assertEqual([row["raw"] for row in filtered["a"]], [unique])
        self.assertEqual(filtered["b"], [])

    def test_reasoning_blend_includes_direct_text_structure(self):
        row = {
            "causal": 0.0,
            "nuance": 0.0,
            "connections": 0.0,
            "problem_solving": 0.0,
            "metacognition": 0.0,
            "reasoning_markers": 2.0,
            "question_quality": 2.0,
        }
        self.assertAlmostEqual(persona_iq._group_scores(row)["reasoning"], 1.3)
        row["llm_reasoning"] = -1.0
        self.assertAlmostEqual(
            persona_iq._group_scores(row)["reasoning"],
            (0.55 * 1.3) + (0.45 * -1.0),
        )

    def test_syntax_requires_structure_not_one_short_marker(self):
        make = lambda text: {
            "raw": text,
            "clean": text,
            "tokens": persona_iq._tokens(text),
        }
        short = persona_iq._interpretable_scored(
            [make("liverpool has not lost though")], lambda _word: 1.0
        )[0]["syntax_peak"][0][0]
        structured = persona_iq._interpretable_scored(
            [make("although the sample is small the result changes because the selection is biased")],
            lambda _word: 1.0,
        )[0]["syntax_peak"][0][0]
        self.assertEqual(short, 0.0)
        self.assertGreater(structured, short)

    def test_breadth_and_depth_are_topic_geometry_only(self):
        row = {
            "topic_breadth": 1.25,
            "niche_depth": -0.75,
            "question_quality": 9.0,
            "technical": 9.0,
            "vocab_peak": 9.0,
        }
        groups = persona_iq._group_scores(row)
        self.assertEqual(groups["breadth"], 1.25)
        self.assertEqual(groups["depth"], -0.75)

    def test_judge_cache_reuses_identical_evidence(self):
        original_path = persona_iq.JUDGE_CACHE
        original_payload = persona_iq._judge_cache_payload
        original_chat = persona_iq._chat_sync
        row = {
            "raw": "because the sample changed the conclusion after selection",
            "clean": "because the sample changed the conclusion after selection",
            "tokens": persona_iq._tokens(
                "because the sample changed the conclusion after selection"
            ),
        }
        try:
            with tempfile.TemporaryDirectory() as directory:
                persona_iq.JUDGE_CACHE = str(Path(directory) / "judge.pkl")
                persona_iq._judge_cache_payload = None
                persona_iq._chat_sync = lambda _prompt: (
                    '{"items":[{"reasoning":3,"abstraction":2,'
                    '"precision":3,"nuance":2,"authored_chat":4}]}'
                )
                first = persona_iq._judge_author("mainuser", [row], items=1)
                persona_iq._chat_sync = lambda _prompt: (_ for _ in ()).throw(
                    AssertionError("cache miss")
                )
                second = persona_iq._judge_author("mainuser", [row], items=1)
        finally:
            persona_iq.JUDGE_CACHE = original_path
            persona_iq._judge_cache_payload = original_payload
            persona_iq._chat_sync = original_chat
        self.assertFalse(first[2])
        self.assertTrue(second[2])


class UserProfilesPureTests(unittest.TestCase):
    def test_profile_candidate_gate_requires_an_explicit_self_value(self):
        accepted = [
            ("location", "i live in poland"),
            ("age", "i'm 25 years old"),
            ("gender", "i'm a woman"),
            ("occupation", "i work as a software developer"),
            ("hobbies", "i've been playing quake"),
            ("languages", "i speak polish"),
        ]
        rejected = [
            ("location", "the cia ruined my country"),
            ("location", "i live in the middle of nowhere"),
            ("location", "i live in the other side of the country"),
            ("age", "this remake is 25 years old"),
            ("gender", "you play as a woman"),
            ("occupation", "why is he in uni"),
            ("occupation", "my boss is annoying"),
            ("occupation", "i work in that field"),
            ("hobbies", "i've been playing for two weeks"),
            ("languages", "if i speak"),
            ("languages", "i'm learning history from video games"),
        ]
        for slot, text in accepted:
            with self.subTest(slot=slot, text=text):
                self.assertTrue(user_profiles._candidate_has_explicit_value(slot, text))
        for slot, text in rejected:
            with self.subTest(slot=slot, text=text):
                self.assertFalse(user_profiles._candidate_has_explicit_value(slot, text))

    def test_incomplete_profile_build_keeps_live_artifact_untouched(self):
        original_candidates = user_profiles.candidate_rows
        row = {
            "id": 1,
            "sent_at": "2026-07-16 01:00:00",
            "channel": "mainroom",
            "content": "i live in poland",
        }
        try:
            user_profiles.candidate_rows = lambda _author, slot, **_kwargs: (
                [row] if slot == "location" else []
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "profiles.json"
                path.write_text('{"sentinel":true}', encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "covered 0/1"):
                    user_profiles.build_profiles(
                        ["mainuser"],
                        llm_call=lambda _system, _user: None,
                        path=path,
                    )
                self.assertEqual(
                    json.loads(path.read_text(encoding="utf-8")),
                    {"sentinel": True},
                )
                self.assertTrue(user_profiles._partial_path(path).exists())
        finally:
            user_profiles.candidate_rows = original_candidates

    def test_artifact_status_rejects_empty_profile_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps({
                "_meta": {
                    "version": user_profiles.VERSION,
                    "alias_signature": chat_archive.alias_signature(),
                },
                "profiles": {f"user{i}": {} for i in range(10)},
                "judged": {},
            }), encoding="utf-8")
            original = artifact_status.USER_PROFILES
            try:
                artifact_status.USER_PROFILES = path
                row = artifact_status._user_profiles_status()
            finally:
                artifact_status.USER_PROFILES = original
        self.assertEqual(row["status"], "warn")
        self.assertIn("all profile records are empty", row["detail"])

    def test_candidate_spread_uses_canonical_author(self):
        class Result:
            def __init__(self, rows):
                self.rows = rows

            def fetchone(self):
                return self.rows[0] if self.rows else None

            def fetchall(self):
                return list(self.rows)

        class FakeConnection:
            def execute(self, query, _params):
                if "MIN(id)" in query:
                    return Result([(1, 100)])
                return Result([(
                    50,
                    "2026-07-15 12:00:00",
                    "mainroom",
                    "i live in a city near the coast",
                )])

        original_connect = user_profiles.chat_archive.connect
        original_echo = user_profiles._said_by_others
        try:
            user_profiles.chat_archive.connect = lambda: FakeConnection()
            user_profiles._said_by_others = lambda _content, _author: False
            rows = user_profiles.candidate_rows(
                "oldalt", "location", per_anchor=4, cap=3
            )
        finally:
            user_profiles.chat_archive.connect = original_connect
            user_profiles._said_by_others = original_echo
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], 50)

    def test_copypasta_query_excludes_entire_target_alias_group(self):
        captured = {}

        class EmptyCursor:
            def fetchall(self):
                return []

        class FakeConnection:
            def execute(self, query, params):
                captured["query"] = query
                captured["params"] = list(params)
                return EmptyCursor()

        original = user_profiles.chat_archive.connect
        try:
            user_profiles.chat_archive.connect = lambda: FakeConnection()
            self.assertFalse(user_profiles._said_by_others(
                "this is one sufficiently long copied personal claim", "oldalt"
            ))
        finally:
            user_profiles.chat_archive.connect = original
        self.assertIn("m.author NOT IN", captured["query"])
        self.assertIn("mainuser", captured["params"])
        self.assertIn("oldalt", captured["params"])

    def _judged(self, value, sent_at, sincerity="sincere", asserts=True):
        # distinct phrasing per row: verbatim repeats are deduped by design
        # (a running gag repeated word-for-word must not corroborate itself)
        return {"asserts": asserts, "value": value, "sincerity": sincerity,
                "plausibility": "ordinary",
                "id": 1, "sent_at": sent_at, "channel": "mainroom",
                "content": f"i said {value} at {sent_at}"}

    def test_parse_judgment_tolerates_prose_around_json(self):
        out = user_profiles._parse_judgment(
            'Sure! Here is the answer:\n{"asserts": true, "value": "Poland", '
            '"sincerity": "sincere"} hope that helps')
        self.assertEqual(out, {"asserts": True, "value": "poland",
                               "sincerity": "sincere", "plausibility": "unclear"})
        self.assertIsNone(user_profiles._parse_judgment("no json here"))
        self.assertIsNone(user_profiles._parse_judgment(None))
        self.assertFalse(user_profiles._normalize_judgment({
            "asserts": "false", "value": None, "sincerity": "unclear"
        })["asserts"])

    def test_profile_build_discards_stale_semantic_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(
                '{"_meta":{"version":2},"profiles":{"mainuser":{}},'
                '"judged":{"old":"verdict"}}',
                encoding="utf-8",
            )
            store = user_profiles.build_profiles(
                [], llm_call=lambda _system, _user: None, path=path
            )
        self.assertEqual(store["judged"], {})
        self.assertEqual(store["profiles"], {})
        self.assertEqual(store["_meta"]["version"], user_profiles.VERSION)

    def test_profile_checkpoint_is_resume_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            store = {
                "_meta": {},
                "profiles": {"mainuser": {"location": {"value": "poland"}}},
                "judged": {f"v{user_profiles.VERSION}|mainuser|location|1": {"asserts": True}},
            }
            user_profiles._checkpoint(store, path)
            loaded = user_profiles.load(path)
            self.assertEqual(loaded["_meta"]["version"], user_profiles.VERSION)
            self.assertEqual(
                loaded["_meta"]["alias_signature"], chat_archive.alias_signature()
            )
            self.assertIn(
                f"v{user_profiles.VERSION}|mainuser|location|1",
                loaded["judged"],
            )

    def test_batch_judge_maps_each_result_back_to_its_receipt(self):
        rows = [
            {"id": 10, "sent_at": "2026-01-01 10:00:00", "channel": "mainroom",
             "content": "i am from poland"},
            {"id": 11, "sent_at": "2026-01-02 10:00:00", "channel": "mainroom",
             "content": "my country is canada Kappa"},
        ]
        response = (
            '{"items":['
            '{"index":1,"asserts":true,"value":"poland",'
            '"sincerity":"sincere","plausibility":"ordinary"},'
            '{"index":2,"asserts":false,"value":null,'
            '"sincerity":"joke","plausibility":"unclear"}'
            ']}'
        )
        judged = user_profiles.judge_candidates_batch(
            "mainuser", "location", rows, lambda _system, _user: response
        )
        self.assertEqual(judged[10]["value"], "poland")
        self.assertFalse(judged[11]["asserts"])
        self.assertEqual(judged[10]["content"], rows[0]["content"])

    def test_reconcile_confirms_only_with_multi_day_support(self):
        one_day = [self._judged("poland", "2026-01-01 10:00:00"),
                   self._judged("poland", "2026-01-01 11:00:00")]
        self.assertEqual(user_profiles._reconcile("location", one_day)["status"],
                         "candidate")
        two_days = one_day + [self._judged("poland", "2026-02-01 10:00:00")]
        self.assertEqual(user_profiles._reconcile("location", two_days)["status"],
                         "confirmed")

    def test_unusual_claim_needs_extra_corroboration(self):
        rows = [
            dict(self._judged("two wives", f"2026-01-0{day} 10:00:00"),
                 plausibility="unusual")
            for day in (1, 2, 3)
        ]
        self.assertEqual(user_profiles._reconcile("relationship", rows[:2])["status"],
                         "candidate")
        self.assertEqual(user_profiles._reconcile("relationship", rows)["status"],
                         "confirmed")

    def test_one_impossible_read_blocks_auto_confirmation(self):
        rows = [
            self._judged("140", "2026-01-01 10:00:00"),
            self._judged("140", "2026-02-01 10:00:00"),
            self._judged("140", "2026-03-01 10:00:00"),
        ]
        rows[-1]["plausibility"] = "impossible"
        self.assertEqual(user_profiles._reconcile("age", rows)["status"], "candidate")

    def test_claim_parser_marks_multiple_spouses_unusual(self):
        claims = user_profiles.claims_in_text("I have two wives")
        self.assertEqual(claims, [{
            "slot": "relationship",
            "value": "2 wives",
            "plausibility": "unusual",
        }])

    def test_irony_format_names_evidence_without_overclaiming(self):
        out = irony.format_analysis({
            "verdict": "likely a repeated bit",
            "confidence": "medium",
            "sarcasm": -0.1,
            "extremity": 0.2,
            "reasons": ["near-copy used by 4 other chatters"],
        })
        self.assertIn("near-copy used by 4 other chatters", out)
        self.assertIn("surface sarcasm -0.1", out)

    def test_placeholder_and_anecdote_values_rejected(self):
        judged = [self._judged("my country", "2026-01-01 10:00:00"),
                  self._judged("my country", "2026-01-02 10:00:00")]
        self.assertIsNone(user_profiles._reconcile("location", judged))
        self.assertFalse(user_profiles._valid_value("i lift way more than 5 lbs"))
        self.assertFalse(user_profiles._valid_value("still working on july 2nd at work"))
        self.assertTrue(user_profiles._valid_value("germany"))
        self.assertTrue(user_profiles._valid_value("software developer"))
        self.assertFalse(user_profiles._valid_value("room next door", slot="location"))
        self.assertFalse(user_profiles._valid_value("peasants since 2000 years", slot="family"))
        self.assertEqual(user_profiles._norm_value("pets", "cat has covid"), "cat")
        self.assertEqual(user_profiles._norm_value("gender", "a girl"), "female")
        self.assertEqual(user_profiles._norm_value("relationship", "two wives"), "2 wives")

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

    def test_candidate_reranker_uses_nonlinear_context_fit(self):
        original = persona_llm._voice_score
        persona_llm._voice_score = lambda _author, _text: (0.5, {})
        try:
            weak, weak_parts = persona_llm._candidate_score(
                "mainuser", "idk maybe", ["favorite", "game"], ["idk maybe"]
            )
            strong, strong_parts = persona_llm._candidate_score(
                "mainuser", "my favorite game is quake", ["favorite", "game"],
                ["idk maybe"],
            )
        finally:
            persona_llm._voice_score = original
        self.assertGreater(strong, weak)
        self.assertGreater(strong_parts["context"], weak_parts["context"])

    def test_candidate_reranker_prefers_target_voice_when_context_ties(self):
        original = persona_llm._voice_score
        persona_llm._voice_score = lambda _author, text: (
            (0.9 if "targetlike" in text else 0.1), {}
        )
        try:
            target, _ = persona_llm._candidate_score(
                "mainuser", "targetlike reply", ["reply"], ["short reply"]
            )
            generic, _ = persona_llm._candidate_score(
                "mainuser", "generic reply", ["reply"], ["short reply"]
            )
        finally:
            persona_llm._voice_score = original
        self.assertGreater(target, generic)


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
    def test_prompt_evidence_dedupes_punctuation_and_case_variants(self):
        rows = persona_llm._unique_messages([
            "This model predicts the same result",
            "this model predicts the same result!",
            "The counterexample changes it",
        ], 5)
        self.assertEqual(rows, [
            "This model predicts the same result",
            "The counterexample changes it",
        ])

    def test_person_vectors_require_matching_provenance(self):
        current = {
            "unit": "utterance",
            "model": "text-embedding-bge-m3",
            "alias_signature": chat_archive.alias_signature(),
            "utterance_version": chat_archive.UTTERANCE_VERSION,
            "vectors": {"mainuser": [1.0]},
        }
        self.assertTrue(persona_embeddings._metadata_current(current))
        current["utterance_version"] -= 1
        self.assertFalse(persona_embeddings._metadata_current(current))

    def test_message_index_metadata_must_match_runtime_provenance(self):
        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        current = {
            "unit": Scalar("utterance"),
            "model": Scalar("text-embedding-bge-m3"),
            "alias_signature": Scalar(chat_archive.alias_signature()),
            "utterance_version": Scalar(chat_archive.UTTERANCE_VERSION),
        }
        self.assertTrue(persona_msg_index._metadata_current(current))
        current.pop("utterance_version")
        self.assertFalse(persona_msg_index._metadata_current(current))

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
