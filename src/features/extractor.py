# src/features/extractor.py
import re
from typing import Dict, Any

def extract_features(text: str) -> Dict[str, Any]:
    """
    Lightweight feature extraction — no heavy models.
    Returns 20+ features usable as scoring context.
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    avg_sentence_len = round(word_count / sentence_count, 2)
    avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 2)

    # Lexical richness
    unique_words = set(w.lower() for w in words)
    vocabulary_richness = round(len(unique_words) / max(word_count, 1), 2)

    # Punctuation signals
    question_count = text.count("?")
    exclamation_count = text.count("!")

    # Hedge words (uncertainty)
    hedge_words = ["maybe", "perhaps", "possibly", "might", "could", "probably",
                   "i think", "i believe", "not sure", "uncertain"]
    hedge_count = sum(1 for h in hedge_words if h in text.lower())

    # Certainty words
    certainty_words = ["definitely", "certainly", "always", "never", "absolutely",
                       "clearly", "obviously", "must", "will"]
    certainty_count = sum(1 for c in certainty_words if c in text.lower())

    # Person references
    first_person = len(re.findall(r'\b(i|me|my|myself|we|our)\b', text.lower()))
    second_person = len(re.findall(r'\b(you|your|yourself)\b', text.lower()))

    # Structural signals
    has_list = bool(re.search(r'(\n[-*•]|\n\d+\.)', text))
    has_code = "```" in text or "`" in text
    has_url = bool(re.search(r'https?://', text))
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])

    # Emotional signals
    positive_words = ["happy", "great", "wonderful", "excellent", "good", "love",
                      "joy", "glad", "pleased", "fantastic", "amazing", "thank"]
    negative_words = ["sad", "angry", "frustrated", "worried", "anxious",
                      "depressed", "upset", "terrible", "awful", "hate", "fear"]
    toxic_words = ["stupid", "idiot", "kill", "hate", "damn", "useless"]

    positive_count = sum(1 for w in positive_words if w in text.lower())
    negative_count = sum(1 for w in negative_words if w in text.lower())
    toxic_count = sum(1 for w in toxic_words if w in text.lower())

    # Readability proxy
    complex_words = [w for w in words if len(w) > 8]
    complexity_ratio = round(len(complex_words) / max(word_count, 1), 2)

    return {
        # Length features
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "avg_word_len": avg_word_len,
        "paragraph_count": paragraph_count,

        # Lexical features
        "vocabulary_richness": vocabulary_richness,
        "complexity_ratio": complexity_ratio,
        "unique_word_count": len(unique_words),

        # Punctuation
        "question_count": question_count,
        "exclamation_count": exclamation_count,

        # Stance
        "hedge_count": hedge_count,
        "certainty_count": certainty_count,
        "first_person_count": first_person,
        "second_person_count": second_person,

        # Structure
        "has_list": has_list,
        "has_code": has_code,
        "has_url": has_url,

        # Emotion signals
        "positive_word_count": positive_count,
        "negative_word_count": negative_count,
        "toxic_word_count": toxic_count,
        "emotion_valence": "positive" if positive_count > negative_count
                          else "negative" if negative_count > positive_count
                          else "neutral",

        # Derived
        "is_short": word_count < 15,
        "is_long": word_count > 80,
        "is_question": question_count > 0,
    }


if __name__ == "__main__":
    test = "I've been feeling really anxious lately. Maybe I should try meditation? I think it could help, but I'm not sure where to start."
    features = extract_features(test)
    for k, v in features.items():
        print(f"  {k}: {v}")