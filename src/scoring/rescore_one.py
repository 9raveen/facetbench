# scripts/rescore_one.py
import json, sys, time
sys.path.insert(0, ".")

from src.scoring.evaluator import score_conversation

with open("data/facets/facets.json") as f:
    facets = json.load(f)

with open("data/conversations/raw/conv_001.json") as f:
    conv = json.load(f)

print("Rescoring conv_001 with calibrated prompts...")
print("This will take ~15-20 min on CPU...")

result = score_conversation(conv, facets)

with open("data/conversations/scored/conv_001_scored.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Overall score: {result['overall_score']}")
print(f"Category averages: {result['category_averages']}")
print(f"Time: {result['processing_time_sec']}s")