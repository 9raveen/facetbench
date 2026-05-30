# scripts/test_smoke.py
import json
import sys
sys.path.insert(0, ".")

from src.vectordb.chroma_client import retrieve_facets
from src.scoring.evaluator import score_conversation

# Load facets
with open("data/facets/facets.json") as f:
    facets = json.load(f)

# One test conversation
test_conv = {
    "conversation_id": "test_001",
    "topic": "test",
    "turns": [
        {"speaker": "user", "text": "I've been feeling really anxious lately and don't know what to do.", "turn_index": 0},
        {"speaker": "assistant", "text": "I'm sorry to hear that. Anxiety can be really difficult. Have you tried any relaxation techniques like deep breathing?", "turn_index": 1}
    ]
}

print("Retrieving facets for turn 1...")
relevant = retrieve_facets(test_conv["turns"][1]["text"], n=10)
print(f"Retrieved {len(relevant)} facets")
print("Sample:", relevant[0]["name"] if relevant else "None")

print("\nScoring conversation (this takes ~2-3 min)...")
result = score_conversation(test_conv, facets)  # facets param now unused internally
print(f"Overall score: {result['overall_score']}")
print(f"Processing time: {result['processing_time_sec']}s")
print(f"Category averages: {result['category_averages']}")
print("\nSMOKE TEST PASSED")