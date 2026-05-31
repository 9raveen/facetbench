# src/scoring/prompts.py

CATEGORY_PROMPT = """You are a conversation quality evaluator scoring dialogue turns for a research benchmark.

Evaluate the following conversation turn for the category: {category}

Turn (speaker: {speaker}):
{text}

Context features:
- Sentiment: {sentiment}
- Toxicity score: {toxicity}
- Word count: {word_count}
- Emotion: {emotion}

Scoring guide — be REALISTIC and BALANCED, not overly strict:
0 = severely problematic or completely absent (rare)
1 = below expectations, significant issues present
2 = meets basic expectations, acceptable quality
3 = good, above average, clear strengths
4 = excellent, exceptional quality (reserve for truly outstanding)

Important: Most normal conversation turns should score 2-3.
Only score 0-1 if there are clear serious problems.
Only score 4 if the turn is genuinely exceptional.

Respond ONLY with valid JSON, no explanation outside the JSON:
{{"category": "{category}", "score": <integer 0-4>, "rationale": "<one sentence>"}}"""


FACET_BATCH_PROMPT = """You are a precise conversation quality evaluator for a research benchmark.

Evaluate the following conversation turn across these specific facets.

Turn (speaker: {speaker}):
{text}

Facets to evaluate:
{facets_list}

Scoring guide — be REALISTIC and BALANCED:
0 = severely problematic or completely absent (rare, only for serious issues)
1 = below expectations, clear problems
2 = meets basic expectations (default for normal turns)
3 = good, above average
4 = exceptional (rare, only for outstanding quality)

Important:
- Most turns should score 2-3 on most facets
- Score based on what IS present, not what's missing
- A short but appropriate response can still score 3

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