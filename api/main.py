# api/main.py
import json
import os
import sys
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI(
    title="FacetBench API",
    description="Production-ready conversation scoring across 300+ facets",
    version="1.0.0"
)

# ── Load facets once at startup ──
with open("data/facets/facets.json") as f:
    FACETS = json.load(f)

FACETS_BY_ID = {f["facet_id"]: f for f in FACETS}

# ── Schemas ──

class Turn(BaseModel):
    speaker: str
    text: str
    turn_index: Optional[int] = 0

class ConversationRequest(BaseModel):
    conversation_id: str
    topic: Optional[str] = ""
    turns: List[Turn]
    mode: Optional[str] = "synthetic"  # "synthetic" | "fast" | "full" | "langgraph"

class ConversationScore(BaseModel):
    conversation_id: str
    topic: str
    total_turns: int
    overall_score: float
    category_averages: Dict[str, float]
    processing_time_sec: float
    model_used: str
    facets_scored: int
    scoring_mode: str

# ── Endpoints ──

@app.get("/health")
def health():
    return {
        "status": "ok",
        "facets_loaded": len(FACETS),
        "model": "qwen2.5:7b",
        "version": "1.0.0"
    }

@app.get("/facets")
def list_facets(
    category: Optional[str] = None,
    group: Optional[str] = None,
    limit: int = 50
):
    facets = FACETS
    if category:
        facets = [f for f in facets if f["category"] == category]
    if group:
        facets = [f for f in facets if f["group"] == group]
    return {
        "count": len(facets),
        "facets": facets[:limit]
    }

@app.get("/facets/{facet_id}")
def get_facet(facet_id: str):
    facet = FACETS_BY_ID.get(facet_id)
    if not facet:
        raise HTTPException(status_code=404, detail=f"Facet {facet_id} not found")
    return facet

@app.get("/facets/categories/summary")
def categories_summary():
    from collections import Counter
    cats = Counter(f["category"] for f in FACETS)
    groups = Counter(f["group"] for f in FACETS)
    return {
        "categories": dict(cats),
        "total_facets": len(FACETS),
        "total_groups": len(groups)
    }

@app.post("/score", response_model=ConversationScore)
def score_conversation_endpoint(request: ConversationRequest):
    conv_dict = {
        "conversation_id": request.conversation_id,
        "topic": request.topic,
        "turns": [
            {"speaker": t.speaker, "text": t.text, "turn_index": t.turn_index}
            for t in request.turns
        ]
    }

    mode = request.mode

    # In deployed environments without Ollama, force synthetic
    if mode in ["full", "fast", "langgraph"]:
        try:
            import httpx as _httpx
            _httpx.get("http://localhost:11434/api/tags", timeout=2)
        except:
            mode = "synthetic"

    if mode == "full":
        from src.scoring.evaluator import score_conversation
        result = score_conversation(conv_dict, FACETS)
    elif mode == "fast":
        from src.scoring.evaluator import score_conversation_fast
        result = score_conversation_fast(conv_dict)
    elif mode == "langgraph":
        from src.pipeline.graph import run_langgraph_pipeline
        result = run_langgraph_pipeline(conv_dict)
    else:
        from src.scoring.synthetic_scorer import score_conversation_synthetic
        result = score_conversation_synthetic(conv_dict, FACETS)

    return ConversationScore(
        conversation_id=result["conversation_id"],
        topic=result.get("topic", ""),
        total_turns=result["total_turns"],
        overall_score=result["overall_score"],
        category_averages=result["category_averages"],
        processing_time_sec=result["processing_time_sec"],
        model_used=result["model_used"],
        facets_scored=result["facets_scored"],
        scoring_mode=result.get("scoring_mode", mode)
    )

@app.get("/benchmark/results")
def benchmark_results():
    scored_dir = "data/conversations/scored"
    if not os.path.exists(scored_dir):
        return {"error": "No scored conversations yet"}

    results = []
    for fname in sorted(os.listdir(scored_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(scored_dir, fname)) as f:
            data = json.load(f)
        if "error" not in data:
            results.append({
                "conversation_id": data.get("conversation_id"),
                "topic": data.get("topic", ""),
                "overall_score": data.get("overall_score"),
                "category_averages": data.get("category_averages"),
                "scoring_mode": data.get("scoring_mode", "unknown"),
                "facets_scored": data.get("facets_scored", 0)
            })

    if not results:
        return {"count": 0, "results": []}

    avg_overall = round(sum(r["overall_score"] for r in results) / len(results), 2)

    return {
        "count": len(results),
        "average_overall_score": avg_overall,
        "results": results
    }

@app.get("/benchmark/conversation/{conversation_id}")
def get_conversation_result(conversation_id: str):
    scored_dir = "data/conversations/scored"
    for fname in os.listdir(scored_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(scored_dir, fname)) as f:
            data = json.load(f)
        if data.get("conversation_id") == conversation_id:
            return data
    raise HTTPException(status_code=404, detail=f"{conversation_id} not found")

@app.get("/benchmark/summary")
def benchmark_summary():
    """Aggregated statistics across all scored conversations."""
    scored_dir = "data/conversations/scored"
    if not os.path.exists(scored_dir):
        return {"error": "No scored conversations yet"}

    results = []
    for fname in sorted(os.listdir(scored_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(scored_dir, fname)) as f:
            data = json.load(f)
        if "error" not in data and "overall_score" in data:
            results.append(data)

    if not results:
        return {"count": 0}

    overall_scores = [r["overall_score"] for r in results]

    # Per-category stats
    category_stats = {}
    for cat in ["Linguistic Quality", "Pragmatics", "Safety", "Emotion"]:
        scores = [
            r["category_averages"][cat]
            for r in results
            if cat in r.get("category_averages", {})
        ]
        if scores:
            category_stats[cat] = {
                "mean": round(sum(scores) / len(scores), 3),
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
                "std": round(
                    (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5, 3
                )
            }

    # Scoring mode breakdown
    from collections import Counter
    mode_counts = Counter(r.get("scoring_mode", "unknown") for r in results)

    return {
        "total_conversations": len(results),
        "overall": {
            "mean": round(sum(overall_scores) / len(overall_scores), 3),
            "min": round(min(overall_scores), 3),
            "max": round(max(overall_scores), 3),
            "std": round(
                (sum((s - sum(overall_scores)/len(overall_scores))**2
                     for s in overall_scores) / len(overall_scores)) ** 0.5, 3
            )
        },
        "by_category": category_stats,
        "scoring_modes": dict(mode_counts),
        "facets_evaluated": 399
    }