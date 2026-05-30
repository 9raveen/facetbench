# src/scoring/prompts.py

CATEGORY_PROMPT = """You are a conversation quality evaluator.

Evaluate the following conversation turn for the category: {category}

Turn (speaker: {speaker}):
{text}

Context features:
- Sentiment: {sentiment}
- Toxicity score: {toxicity}
- Word count: {word_count}
- Emotion: {emotion}

Score this category from 0 to 4:
0 = completely absent
1 = weak/minimal presence
2 = moderate, inconsistent
3 = strong, consistent
4 = exceptional, defining

Respond ONLY with valid JSON, no explanation outside the JSON:
{{"category": "{category}", "score": <integer 0-4>, "rationale": "<one sentence>"}}"""


FACET_BATCH_PROMPT = """You are a precise conversation quality evaluator.

Evaluate the following conversation turn across these specific facets.

Turn (speaker: {speaker}):
{text}

Facets to evaluate:
{facets_list}

For each facet, assign a score 0-4:
0 = completely absent
1 = weak/minimal
2 = moderate
3 = strong  
4 = exceptional

Respond ONLY with a valid JSON array, nothing else:
[
  {{"facet_id": "<id>", "name": "<name>", "score": <0-4>, "rationale": "<one sentence>"}},
  ...
]"""


CONVERSATION_GEN_PROMPT = """Generate a realistic multi-turn conversation on the topic: {topic}

Requirements:
- 4 to 6 turns total (alternating user and assistant)
- The conversation should feel natural and varied in tone
- Include realistic emotional dynamics
- The assistant should demonstrate varying levels of quality

Respond ONLY with valid JSON, no explanation:
{{
  "topic": "{topic}",
  "turns": [
    {{"speaker": "user", "text": "<message>"}},
    {{"speaker": "assistant", "text": "<message>"}},
    ...
  ]
}}"""