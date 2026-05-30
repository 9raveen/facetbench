# tests/test_pipeline.py
import sys
import json
import pytest
sys.path.insert(0, ".")

from src.features.extractor import extract_features
from src.scoring.synthetic_scorer import score_conversation_synthetic
from src.vectordb.chroma_client import retrieve_facets

# ── Load facets once ──
with open("data/facets/facets.json") as f:
    FACETS = json.load(f)

SAMPLE_CONVERSATION = {
    "conversation_id": "test_001",
    "topic": "mental health",
    "turns": [
        {"speaker": "user", "text": "I have been feeling really anxious lately and cannot sleep.", "turn_index": 0},
        {"speaker": "assistant", "text": "I am sorry to hear that. Anxiety and sleep issues often go together. Have you tried any relaxation techniques?", "turn_index": 1},
        {"speaker": "user", "text": "Not really. I do not know where to start.", "turn_index": 2},
        {"speaker": "assistant", "text": "A good starting point is deep breathing. Try inhaling for 4 counts, holding for 4, exhaling for 4. It activates your parasympathetic nervous system.", "turn_index": 3}
    ]
}


class TestFeatureExtractor:

    def test_returns_dict(self):
        features = extract_features("Hello world, how are you today?")
        assert isinstance(features, dict)

    def test_word_count_correct(self):
        features = extract_features("This is a five word sentence here")
        assert features["word_count"] == 7

    def test_question_detected(self):
        features = extract_features("How are you doing today?")
        assert features["is_question"] is True

    def test_no_question(self):
        features = extract_features("I am doing fine today.")
        assert features["is_question"] is False

    def test_emotion_valence_positive(self):
        features = extract_features("I am so happy and grateful today, everything is wonderful.")
        assert features["emotion_valence"] == "positive"

    def test_emotion_valence_negative(self):
        features = extract_features("I feel sad, angry and frustrated about everything.")
        assert features["emotion_valence"] == "negative"

    def test_all_keys_present(self):
        features = extract_features("Sample text for testing purposes.")
        expected_keys = [
            "word_count", "sentence_count", "avg_sentence_len",
            "vocabulary_richness", "question_count", "hedge_count",
            "has_list", "has_code", "emotion_valence", "is_short", "is_long"
        ]
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"

    def test_short_text_detected(self):
        features = extract_features("Hi there.")
        assert features["is_short"] is True

    def test_long_text_detected(self):
        long_text = " ".join(["word"] * 100)
        features = extract_features(long_text)
        assert features["is_long"] is True

    def test_code_detected(self):
        features = extract_features("Here is some code: ```python print('hello')```")
        assert features["has_code"] is True


class TestSyntheticScorer:

    def test_returns_dict(self):
        result = score_conversation_synthetic(SAMPLE_CONVERSATION, FACETS)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = score_conversation_synthetic(SAMPLE_CONVERSATION, FACETS)
        required = [
            "conversation_id", "overall_score", "category_averages",
            "turn_scores", "facets_scored", "scoring_mode"
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_overall_score_in_range(self):
        result = score_conversation_synthetic(SAMPLE_CONVERSATION, FACETS)
        assert 0 <= result["overall_score"] <= 4

    def test_category_averages_present(self):
        result = score_conversation_synthetic(SAMPLE_CONVERSATION, FACETS)
        categories = ["Linguistic Quality", "Pragmatics", "Safety", "Emotion"]
        for cat in categories:
            assert cat in result["category_averages"]

    def test_category_scores_in_range(self):
        result = score_conversation_synthetic(SAMPLE_CONVERSATION, FACETS)
        for cat, score in result["category_averages"].items():
            assert 0 <= score <= 4, f"{cat} score {score} out of range"

    def test_turn_scores_count(self):
        result = score_conversation_synthetic(SAMPLE_CONVERSATION, FACETS)
        assert result["total_turns"] == 4

    def test_facets_scored(self):
        result = score_conversation_synthetic(SAMPLE_CONVERSATION, FACETS)
        assert result["facets_scored"] == 30

    def test_conversation_id_preserved(self):
        result = score_conversation_synthetic(SAMPLE_CONVERSATION, FACETS)
        assert result["conversation_id"] == "test_001"

    def test_short_turns_skipped(self):
        conv = {
            "conversation_id": "test_short",
            "topic": "test",
            "turns": [
                {"speaker": "user", "text": "Hi", "turn_index": 0},
                {"speaker": "assistant", "text": "This is a proper response with enough words to be scored.", "turn_index": 1}
            ]
        }
        result = score_conversation_synthetic(conv, FACETS)
        assert result["total_turns"] == 1


class TestVectorDB:

    def test_retrieval_returns_list(self):
        results = retrieve_facets("I feel anxious and cannot sleep", n=10)
        assert isinstance(results, list)

    def test_retrieval_count(self):
        results = retrieve_facets("technical programming debugging", n=15)
        assert len(results) == 15

    def test_retrieval_has_required_fields(self):
        results = retrieve_facets("happy joyful positive emotion", n=5)
        for facet in results:
            assert "facet_id" in facet
            assert "name" in facet
            assert "category" in facet

    def test_semantic_relevance(self):
        """Safety-related query should return some Safety facets."""
        results = retrieve_facets("dangerous harmful violent threatening content", n=20)
        categories = [f["category"] for f in results]
        assert "Safety" in categories

    def test_different_queries_different_results(self):
        results_a = retrieve_facets("happy joyful emotion positive", n=10)
        results_b = retrieve_facets("dangerous harmful toxic violent", n=10)
        ids_a = set(f["facet_id"] for f in results_a)
        ids_b = set(f["facet_id"] for f in results_b)
        assert ids_a != ids_b