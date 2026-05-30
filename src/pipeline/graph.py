# src/pipeline/graph.py
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from langgraph.graph import StateGraph, END
from src.pipeline.state import ConversationState
from src.pipeline.nodes import (
    preprocess_node,
    feature_extraction_node,
    facet_retrieval_node,
    category_evaluation_node,
    facet_evaluation_node,
    score_aggregation_node,
    output_formatting_node
)

def build_pipeline():
    workflow = StateGraph(ConversationState)

    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("feature_extraction", feature_extraction_node)
    workflow.add_node("facet_retrieval", facet_retrieval_node)
    workflow.add_node("category_evaluation", category_evaluation_node)
    workflow.add_node("facet_evaluation", facet_evaluation_node)
    workflow.add_node("score_aggregation", score_aggregation_node)
    workflow.add_node("output_formatting", output_formatting_node)

    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "feature_extraction")
    workflow.add_edge("feature_extraction", "facet_retrieval")
    workflow.add_edge("facet_retrieval", "category_evaluation")
    workflow.add_edge("category_evaluation", "facet_evaluation")
    workflow.add_edge("facet_evaluation", "score_aggregation")
    workflow.add_edge("score_aggregation", "output_formatting")
    workflow.add_edge("output_formatting", END)

    return workflow.compile()

pipeline = build_pipeline()

def run_langgraph_pipeline(conversation: dict) -> dict:
    """
    Entry point for FastAPI.
    Takes conversation dict, runs full LangGraph pipeline,
    returns standardized score output.
    """
    start = time.time()

    initial_state = ConversationState(
        conversation_id=conversation.get("conversation_id", "unknown"),
        raw_turns=conversation.get("turns", []),
        processed_turns=[],
        relevant_facets={},
        category_scores={},
        facet_scores={},
        final_scores={},
        errors=[],
        processing_time_sec=0.0
    )

    final_state = pipeline.invoke(initial_state)

    elapsed = round(time.time() - start, 2)
    final_scores = final_state.get("final_scores", {})

    # Build standardized output matching other scoring modes
    turn_scores = []
    for turn in final_state.get("processed_turns", []):
        idx = turn["turn_index"]
        turn_scores.append({
            "turn_index": idx,
            "speaker": turn["speaker"],
            "text_preview": turn["text"][:100],
            "category_scores": final_state.get("category_scores", {}).get(idx, {}),
            "facet_scores": final_state.get("facet_scores", {}).get(idx, [])
        })

    return {
        "conversation_id": conversation.get("conversation_id", "unknown"),
        "topic": conversation.get("topic", ""),
        "total_turns": len(turn_scores),
        "overall_score": final_scores.get("overall", 0.0),
        "category_averages": final_scores.get("category_averages", {}),
        "turn_scores": turn_scores,
        "processing_time_sec": elapsed,
        "model_used": "qwen2.5:7b",
        "facets_scored": final_scores.get("facets_scored", 0),
        "scoring_mode": "langgraph"
    }

if __name__ == "__main__":
    print("LangGraph pipeline compiled successfully")
    print("Nodes:", list(pipeline.nodes.keys()) if hasattr(pipeline, 'nodes') else "compiled")