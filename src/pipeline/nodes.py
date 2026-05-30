# src/pipeline/nodes.py
from src.pipeline.state import ConversationState
from src.vectordb.chroma_client import retrieve_facets
from src.scoring.evaluator import score_category, score_with_confidence, chunk, CATEGORIES
from src.scoring.synthetic_scorer import analyze_text_heuristics
import time

def preprocess_node(state: ConversationState) -> ConversationState:
    """Clean text, assign turn indices, normalize speakers."""
    processed = []
    for i, turn in enumerate(state["raw_turns"]):
        processed.append({
            **turn,
            "turn_index": i,
            "text_clean": turn["text"].strip(),
            "word_count": len(turn["text"].split())
        })
    state["processed_turns"] = processed
    return state

def feature_extraction_node(state: ConversationState) -> ConversationState:
    """Extract heuristic features per turn."""
    for turn in state["processed_turns"]:
        turn["features"] = analyze_text_heuristics(turn["text_clean"])
    return state

def facet_retrieval_node(state: ConversationState) -> ConversationState:
    """Retrieve top-30 relevant facets per turn from ChromaDB."""
    relevant = {}
    for turn in state["processed_turns"]:
        if turn["word_count"] < 3:
            continue
        facets = retrieve_facets(turn["text_clean"], n=30)
        relevant[turn["turn_index"]] = facets
    state["relevant_facets"] = relevant
    return state

def category_evaluation_node(state: ConversationState) -> ConversationState:
    """Level 1: Score 4 categories per turn."""
    scores = {}
    for turn in state["processed_turns"]:
        if turn["word_count"] < 3:
            continue
        turn_scores = {}
        for cat in CATEGORIES:
            result = score_category(turn, cat)
            turn_scores[cat] = {
                "score": result.get("score", 2),
                "rationale": result.get("rationale", "")
            }
        scores[turn["turn_index"]] = turn_scores
    state["category_scores"] = scores
    return state

def facet_evaluation_node(state: ConversationState) -> ConversationState:
    """Level 3: Score facets in batches of 10."""
    facet_scores = {}
    for turn in state["processed_turns"]:
        idx = turn["turn_index"]
        facets = state["relevant_facets"].get(idx, [])
        if not facets:
            continue
        batches = chunk(facets, 10)
        scores = []
        for batch in batches:
            scores.extend(score_with_confidence(turn, batch, n=1))
        facet_scores[idx] = scores
    state["facet_scores"] = facet_scores
    return state

def score_aggregation_node(state: ConversationState) -> ConversationState:
    """Roll up facet scores to conversation level."""
    all_facet_scores = [
        fs for turn_scores in state["facet_scores"].values()
        for fs in turn_scores
    ]
    overall = round(
        sum(s["score"] for s in all_facet_scores) / max(len(all_facet_scores), 1), 2
    )
    cat_averages = {}
    for cat in CATEGORIES:
        cat_scores = [
            state["category_scores"].get(idx, {}).get(cat, {}).get("score", 2)
            for idx in state["category_scores"]
        ]
        cat_averages[cat] = round(sum(cat_scores) / max(len(cat_scores), 1), 2)

    state["final_scores"] = {
        "overall": overall,
        "category_averages": cat_averages,
        "facets_scored": len(set(fs["facet_id"] for fs in all_facet_scores))
    }
    return state

def output_formatting_node(state: ConversationState) -> ConversationState:
    """Format final output."""
    state["final_scores"]["conversation_id"] = state["conversation_id"]
    state["final_scores"]["model"] = "qwen2.5:7b"
    return state