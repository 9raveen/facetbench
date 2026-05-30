# src/pipeline/graph.py
# src/pipeline/graph.py
import sys
import os
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

# Build at import time
pipeline = build_pipeline()

if __name__ == "__main__":
    print("LangGraph pipeline compiled successfully")
    print("Nodes:", list(pipeline.nodes.keys()) if hasattr(pipeline, 'nodes') else "compiled")