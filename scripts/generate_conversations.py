# scripts/generate_conversations.py
import json
import sys
import os
sys.path.insert(0, ".")

from src.scoring.evaluator import llm, extract_json
from src.scoring.prompts import CONVERSATION_GEN_PROMPT

TOPICS = [
    "mental health support and anxiety",
    "technical debugging a Python error",
    "customer complaint about a product",
    "career advice for a student",
    "relationship conflict resolution",
    "explaining a complex science concept",
    "financial planning for beginners",
    "cooking recipe guidance",
    "travel planning recommendations",
    "job interview preparation",
    "dealing with grief and loss",
    "learning a new programming language",
    "home workout and fitness advice",
    "managing work-life balance",
    "understanding a legal document",
    "climate change discussion",
    "parenting advice",
    "negotiating a salary raise",
    "dealing with a difficult coworker",
    "starting a small business",
    "academic essay writing help",
    "medical symptom inquiry",
    "political debate on social media",
    "creative writing assistance",
    "learning meditation and mindfulness",
    "gaming strategy advice",
    "home repair guidance",
    "personal finance budgeting",
    "relationship advice for teens",
    "coping with job loss",
    "language learning tips",
    "nutrition and diet planning",
    "conflict between friends",
    "college application essay help",
    "dealing with insomnia",
    "choosing a laptop for work",
    "understanding cryptocurrency",
    "pet care advice",
    "managing social anxiety",
    "ethical dilemma discussion",
    "sports performance improvement",
    "creative block in art",
    "elder care planning",
    "online privacy and security",
    "dealing with academic failure",
    "understanding insurance policies",
    "environmental activism discussion",
    "workplace discrimination concern",
    "learning to drive advice",
    "moving to a new city alone"
]

os.makedirs("data/conversations/raw", exist_ok=True)

def generate_one(topic: str, idx: int) -> dict:
    prompt = CONVERSATION_GEN_PROMPT.format(topic=topic)
    response = llm.invoke(prompt)
    result = extract_json(response)
    
    if not result or "turns" not in result:
        # Fallback hardcoded conversation if Qwen fails to parse
        result = {
            "topic": topic,
            "turns": [
                {"speaker": "user", "text": f"I need help with {topic}."},
                {"speaker": "assistant", "text": f"I'd be happy to help you with {topic}. Could you tell me more about your specific situation?"},
                {"speaker": "user", "text": "I've been struggling with this for a while and don't know where to start."},
                {"speaker": "assistant", "text": "That's completely understandable. Let's break this down step by step so it feels more manageable."}
            ]
        }
    
    # Add metadata
    result["conversation_id"] = f"conv_{idx:03d}"
    result["topic"] = topic
    for i, turn in enumerate(result["turns"]):
        turn["turn_index"] = i
    
    return result

if __name__ == "__main__":
    print(f"Generating {len(TOPICS)} conversations...")
    
    for i, topic in enumerate(TOPICS):
        print(f"[{i+1}/{len(TOPICS)}] {topic}...", end=" ", flush=True)
        try:
            conv = generate_one(topic, i+1)
            path = f"data/conversations/raw/conv_{i+1:03d}.json"
            with open(path, "w") as f:
                json.dump(conv, f, indent=2)
            print(f"✓ {len(conv['turns'])} turns")
        except Exception as e:
            print(f"✗ ERROR: {e}")
    
    print(f"\nDone. Check data/conversations/raw/")