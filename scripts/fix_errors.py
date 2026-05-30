# scripts/fix_errors.py
import json
import os
import sys
sys.path.insert(0, ".")

from src.scoring.synthetic_scorer import score_conversation_synthetic

SCORED_DIR = "data/conversations/scored"
RAW_DIR = "data/conversations/raw"

with open("data/facets/facets.json") as f:
    facets = json.load(f)

fixed = 0
for fname in sorted(os.listdir(SCORED_DIR)):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(SCORED_DIR, fname)
    with open(path) as f:
        data = json.load(f)

    # Check if it's an error file
    if "error" in data or "overall_score" not in data:
        print(f"Fixing {fname}...")
        conv_id = fname.replace("_scored.json", "")
        raw_path = os.path.join(RAW_DIR, f"{conv_id}.json")

        if os.path.exists(raw_path):
            with open(raw_path) as f:
                conv = json.load(f)
    
    # Normalize turn keys — handle 'content' or 'message' instead of 'text'
            for turn in conv.get("turns", []):
                if "text" not in turn:
                    turn["text"] = turn.get("content", turn.get("message", turn.get("response", "")))
    
            result = score_conversation_synthetic(conv, facets)
            result["scoring_mode"] = "synthetic_fallback"
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  ✓ Fixed with synthetic score={result['overall_score']}")
            fixed += 1
        else:
            print(f"  ✗ Raw file not found: {raw_path}")

print(f"\nFixed {fixed} error files.")