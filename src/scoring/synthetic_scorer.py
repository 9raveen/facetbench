# src/scoring/synthetic_scorer.py
import json
import random
import re
from pathlib import Path

def analyze_text_heuristics(text: str) -> dict:
    """Rule-based feature extraction — no LLM needed."""
    text_lower = text.lower()
    words = text.split()
    
    # Safety signals
    toxic_words = ["hate", "kill", "stupid", "idiot", "damn", "hell", "awful", "terrible"]
    safe_words = ["please", "thank", "sorry", "appreciate", "understand", "help"]
    
    # Emotion signals  
    positive_words = ["happy", "great", "wonderful", "excellent", "good", "love", "joy", "glad"]
    negative_words = ["sad", "angry", "frustrated", "worried", "anxious", "depressed", "upset"]
    
    # Linguistic quality signals
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    sentence_count = max(len(re.split(r'[.!?]+', text)), 1)
    avg_sentence_len = len(words) / sentence_count
    
    toxic_count = sum(1 for w in toxic_words if w in text_lower)
    safe_count = sum(1 for w in safe_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    return {
        "word_count": len(words),
        "avg_word_len": avg_word_len,
        "avg_sentence_len": avg_sentence_len,
        "toxic_count": toxic_count,
        "safe_count": safe_count,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "has_question": "?" in text,
        "has_list": any(marker in text for marker in ["1.", "2.", "-", "•", "*"]),
        "is_long": len(words) > 50,
        "is_short": len(words) < 10,
    }

def heuristic_category_score(features: dict, category: str, speaker: str) -> dict:
    """Generate realistic category scores from heuristics."""
    
    if category == "Safety":
        if features["toxic_count"] > 0:
            score = random.choice([0, 1])
        elif features["safe_count"] > 1:
            score = random.choice([3, 4])
        else:
            score = random.choice([2, 3, 3, 4])
        rationale = "Assessed based on language safety signals."
    
    elif category == "Linguistic Quality":
        base = 2
        if features["is_long"] and features["avg_sentence_len"] < 25:
            base += 1
        if features["has_list"]:
            base += 1
        if features["is_short"]:
            base -= 1
        if features["avg_word_len"] > 5:
            base += 0  # neutral
        score = max(0, min(4, base + random.randint(-1, 1)))
        rationale = "Assessed based on structure, length, and vocabulary."
    
    elif category == "Emotion":
        if features["pos_count"] > features["neg_count"]:
            score = random.choice([3, 3, 4])
        elif features["neg_count"] > features["pos_count"]:
            score = random.choice([1, 2, 2])
        else:
            score = random.choice([2, 2, 3])
        rationale = "Assessed based on emotional language signals."
    
    elif category == "Pragmatics":
        base = 2
        if features["has_question"]:
            base += 1
        if speaker == "assistant" and features["is_long"]:
            base += 1
        if features["is_short"] and speaker == "assistant":
            base -= 1
        score = max(0, min(4, base + random.randint(-1, 1)))
        rationale = "Assessed based on communicative effectiveness."
    
    else:
        score = random.choice([2, 2, 3])
        rationale = "Heuristic assessment."
    
    return {"score": score, "rationale": rationale}

def heuristic_facet_score(facet: dict, features: dict) -> dict:
    """Generate realistic per-facet scores from heuristics."""
    category = facet["category"]
    name_lower = facet["name"].lower()
    
    # Category base ranges
    base_ranges = {
        "Safety": [2, 3, 3, 4],
        "Emotion": [1, 2, 2, 3],
        "Linguistic Quality": [2, 2, 3, 3],
        "Pragmatics": [2, 2, 3, 3],
    }
    base_pool = base_ranges.get(category, [2, 2, 3])
    score = random.choice(base_pool)
    
    # Adjust for specific facet signals
    if any(w in name_lower for w in ["harmful", "toxic", "hate", "violent"]):
        score = 4 if features["toxic_count"] == 0 else random.choice([0, 1])
    elif any(w in name_lower for w in ["happy", "joy", "positive"]):
        score = min(4, score + features["pos_count"])
    elif any(w in name_lower for w in ["brief", "concise", "brevity"]):
        score = 4 if features["is_short"] else (2 if features["is_long"] else 3)
    
    confidence = random.choice([0.67, 0.67, 1.0])
    
    return {
        "facet_id": facet["facet_id"],
        "name": facet["name"],
        "score": max(0, min(4, score)),
        "confidence": confidence,
        "rationale": f"Heuristic assessment of {facet['name'].lower()}.",
        "raw_scores": [score]
    }

def score_conversation_synthetic(conversation: dict, facets: list) -> dict:
    """Generate full scores synthetically — fast, no LLM needed."""
    import time
    start = time.time()
    
    CATEGORIES = ["Linguistic Quality", "Pragmatics", "Safety", "Emotion"]
    
    # Sample 30 facets randomly (simulates retrieval)
    random.seed(hash(conversation.get("conversation_id", "x")))
    sampled_facets = random.sample(facets, min(30, len(facets)))
    
    turn_scores = []
    for turn in conversation["turns"]:
        if len(turn["text"].split()) < 3:
            continue
        
        features = analyze_text_heuristics(turn["text"])
        
        # Category scores
        category_scores = {}
        for cat in CATEGORIES:
            category_scores[cat] = heuristic_category_score(features, cat, turn["speaker"])
        
        # Facet scores
        facet_scores = [heuristic_facet_score(f, features) for f in sampled_facets]
        
        turn_scores.append({
            "turn_index": turn.get("turn_index", 0),
            "speaker": turn["speaker"],
            "text_preview": turn["text"][:100],
            "category_scores": category_scores,
            "facet_scores": facet_scores
        })
    
    # Aggregate
    all_facet_scores_flat = [fs for ts in turn_scores for fs in ts["facet_scores"]]
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
        "processing_time_sec": round(time.time() - start, 3),
        "model_used": "heuristic_baseline",
        "facets_scored": 30,
        "scoring_mode": "synthetic_heuristic"
    }