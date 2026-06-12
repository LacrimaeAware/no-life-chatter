import importlib
import math
import sys
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
    )
    sys.modules["config"] = fake
    return fake


install_fake_config()
from utils import chat_archive, persona_classifier  # noqa: E402
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

    def test_alias_cycle_stops_instead_of_looping_forever(self):
        aliases = {"a": "b", "b": "a"}
        self.assertIn(chat_archive._resolve_alias("a", aliases), {"a", "b"})


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


if __name__ == "__main__":
    unittest.main()
