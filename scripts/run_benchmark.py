# scripts/run_benchmark.py
import json, os, sys, time
sys.path.insert(0, ".")

from src.scoring.evaluator import score_conversation, score_conversation_fast
from src.scoring.synthetic_scorer import score_conversation_synthetic

RAW_DIR = "data/conversations/raw"
SCORED_DIR = "data/conversations/scored"
FACETS_PATH = "data/facets/facets.json"

os.makedirs(SCORED_DIR, exist_ok=True)

with open(FACETS_PATH) as f:
    facets = json.load(f)

conv_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".json")])
print(f"Found {len(conv_files)} conversations")
print("conv_001-005 → FULL LLM scoring (deep, overnight)")
print("conv_006-010 → FAST LLM scoring (category only)")
print("conv_011-050 → SYNTHETIC scoring (heuristic, instant)\n")

for i, fname in enumerate(conv_files):
    path = os.path.join(RAW_DIR, fname)
    out_path = os.path.join(SCORED_DIR, fname.replace(".json", "_scored.json"))

    if os.path.exists(out_path):
        print(f"[{i+1}/{len(conv_files)}] {fname} already scored, skipping.")
        continue

    with open(path) as f:
        conv = json.load(f)

    print(f"[{i+1}/{len(conv_files)}] {fname}...", end=" ", flush=True)
    t0 = time.time()

    try:
        if i < 5:
            result = score_conversation(conv, facets)
            mode = "full_llm"
        elif i < 10:
            result = score_conversation_fast(conv)
            mode = "fast_llm"
        else:
            result = score_conversation_synthetic(conv, facets)
            mode = "synthetic"

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"✓ [{mode}] score={result['overall_score']} time={round(time.time()-t0,1)}s")

    except Exception as e:
    print(f"✗ ERROR: {e}")
    # Fallback to synthetic so we always have a score
    try:
        result = score_conversation_synthetic(conv, facets)
        result["scoring_mode"] = "synthetic_fallback"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → Saved synthetic fallback score={result['overall_score']}")
    except Exception as e2:
        with open(out_path, "w") as f:
            json.dump({"conversation_id": conv.get("conversation_id"), "error": str(e)}, f)

print("\nAll done. Check data/conversations/scored/")