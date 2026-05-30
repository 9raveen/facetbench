# src/scoring/evaluator.py
import json
import re
import time
from langchain_ollama import OllamaLLM
from src.scoring.prompts import CATEGORY_PROMPT, FACET_BATCH_PROMPT

MODEL_NAME = "qwen2.5:7b"
CATEGORIES = ["Linguistic Quality", "Pragmatics", "Safety", "Emotion"]

llm = OllamaLLM(model=MODEL_NAME, temperature=0.3)
llm_consistency = OllamaLLM(model=MODEL_NAME, temperature=0.7)

def extract_json(text: str):
    """Robust JSON extractor — handles markdown fences and extra text."""
    text = text.strip()
    # Remove markdown fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    # Find first { or [ and try from there
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
    return None

def get_basic_features(text: str) -> dict:
    """Lightweight features — no heavy models needed."""
    words = text.split()
    return {
        "word_count": len(words),
        "sentiment": "neutral",   # placeholder — VADER optional
        "toxicity": 0.0,          # placeholder — detoxify optional
        "emotion": "neutral"      # placeholder
    }

def score_category(turn: dict, category: str) -> dict:
    features = get_basic_features(turn["text"])
    prompt = CATEGORY_PROMPT.format(
        category=category,
        speaker=turn["speaker"],
        text=turn["text"],
        sentiment=features["sentiment"],
        toxicity=features["toxicity"],
        word_count=features["word_count"],
        emotion=features["emotion"]
    )
    response = llm.invoke(prompt)
    result = extract_json(response)
    if result and "score" in result:
        return result
    return {"category": category, "score": 2, "rationale": "Parse error — defaulting to 2"}

def score_facet_batch(turn: dict, facets: list) -> list:
    """Score a batch of up to 10 facets in one LLM call."""
    facets_list = "\n".join([
        f"{i+1}. [{f['facet_id']}] {f['name']}: {f['description']}"
        for i, f in enumerate(facets)
    ])
    prompt = FACET_BATCH_PROMPT.format(
        speaker=turn["speaker"],
        text=turn["text"],
        facets_list=facets_list
    )
    response = llm.invoke(prompt)
    result = extract_json(response)
    if isinstance(result, list):
        return result
    # Fallback: return default scores so pipeline doesn't crash
    return [
        {"facet_id": f["facet_id"], "name": f["name"], "score": 2, "rationale": "Parse error"}
        for f in facets
    ]

def score_with_confidence(turn: dict, facets: list, n: int = 2) -> list:
    """Run batch scoring N times, use mode score as final + confidence."""
    all_runs = []
    for _ in range(n):
        run = score_facet_batch(turn, facets)
        all_runs.append({r["facet_id"]: r["score"] for r in run})

    # Get first run as base structure
    base = score_facet_batch(turn, facets)

    for item in base:
        fid = item["facet_id"]
        scores = [run.get(fid, 2) for run in all_runs]
        from collections import Counter
        most_common_score, count = Counter(scores).most_common(1)[0]
        item["score"] = most_common_score
        item["confidence"] = round(count / n, 2)
        item["raw_scores"] = scores

    return base

def chunk(lst: list, size: int) -> list:
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def score_conversation(conversation: dict, facets: list) -> dict:
    """
    Main entry point.
    conversation = {"conversation_id": ..., "topic": ..., "turns": [...]}
    facets = full facets list (used for metadata only, retrieval via ChromaDB)
    """
    from src.vectordb.chroma_client import retrieve_facets
    
    start = time.time()
    turn_scores = []

    for turn in conversation["turns"]:
        # Skip very short turns
        if len(turn["text"].split()) < 3:
            continue

        # Level 1: category scores (4 calls)
        category_scores = {}
        for cat in CATEGORIES:
            result = score_category(turn, cat)
            category_scores[cat] = {
                "score": result.get("score", 2),
                "rationale": result.get("rationale", "")
            }

        # Level 3: retrieve top 30 relevant facets, score in batches of 10
        relevant_facets = retrieve_facets(turn["text"], n=30)
        batches = chunk(relevant_facets, 10)
        all_facet_scores = []
        for batch in batches:
            batch_result = score_with_confidence(turn, batch, n=1)
            all_facet_scores.extend(batch_result)

        turn_scores.append({
            "turn_index": turn.get("turn_index", 0),
            "speaker": turn["speaker"],
            "text_preview": turn["text"][:100],
            "category_scores": category_scores,
            "facet_scores": all_facet_scores
        })

    # Aggregate
    all_facet_scores_flat = [
        fs for ts in turn_scores for fs in ts["facet_scores"]
    ]
    overall = round(
        sum(s["score"] for s in all_facet_scores_flat) / max(len(all_facet_scores_flat), 1), 2
    )

    cat_averages = {}
    for cat in CATEGORIES:
        cat_scores = [
            ts["category_scores"][cat]["score"]
            for ts in turn_scores if cat in ts["category_scores"]
        ]
        cat_averages[cat] = round(sum(cat_scores) / max(len(cat_scores), 1), 2)

    return {
        "conversation_id": conversation.get("conversation_id", "unknown"),
        "topic": conversation.get("topic", ""),
        "total_turns": len(turn_scores),
        "overall_score": overall,
        "category_averages": cat_averages,
        "turn_scores": turn_scores,
        "processing_time_sec": round(time.time() - start, 2),
        "model_used": MODEL_NAME,
        "facets_scored": 30
    }
def score_conversation_fast(conversation: dict) -> dict:
    """
    Fast path: category scores only, no facet-level scoring.
    Used for bulk scoring when time is limited.
    ~2 min per conversation on CPU vs ~60 min for full scoring.
    """
    start = time.time()
    turn_scores = []

    for turn in conversation["turns"]:
        if len(turn["text"].split()) < 3:
            continue

        category_scores = {}
        for cat in CATEGORIES:
            result = score_category(turn, cat)
            category_scores[cat] = {
                "score": result.get("score", 2),
                "rationale": result.get("rationale", "")
            }

        turn_scores.append({
            "turn_index": turn.get("turn_index", 0),
            "speaker": turn["speaker"],
            "text_preview": turn["text"][:100],
            "category_scores": category_scores,
            "facet_scores": []  # empty for fast path
        })

    cat_averages = {}
    for cat in CATEGORIES:
        cat_scores = [
            ts["category_scores"][cat]["score"]
            for ts in turn_scores if cat in ts["category_scores"]
        ]
        cat_averages[cat] = round(sum(cat_scores) / max(len(cat_scores), 1), 2)

    overall = round(sum(cat_averages.values()) / len(cat_averages), 2)

    return {
        "conversation_id": conversation.get("conversation_id", "unknown"),
        "topic": conversation.get("topic", ""),
        "total_turns": len(turn_scores),
        "overall_score": overall,
        "category_averages": cat_averages,
        "turn_scores": turn_scores,
        "processing_time_sec": round(time.time() - start, 2),
        "model_used": MODEL_NAME,
        "facets_scored": 0,
        "scoring_mode": "fast"
    }