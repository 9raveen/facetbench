# src/pipeline/state.py
from typing import TypedDict, List, Dict, Any, Optional

class ConversationState(TypedDict):
    # Input
    conversation_id: str
    raw_turns: List[Dict[str, str]]

    # After preprocessing
    processed_turns: List[Dict[str, Any]]

    # After facet retrieval
    relevant_facets: Dict[str, List[Dict]]

    # After category evaluation
    category_scores: Dict[str, Dict]

    # After facet evaluation
    facet_scores: Dict[str, List[Dict]]

    # After aggregation
    final_scores: Dict[str, Any]

    # Control
    errors: List[str]
    processing_time_sec: float