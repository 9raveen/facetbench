import json, sys
sys.path.insert(0, '.')
from src.scoring.synthetic_scorer import score_conversation_synthetic

with open('data/facets/facets.json') as f:
    facets = json.load(f)

with open('data/conversations/raw/conv_005.json') as f:
    conv = json.load(f)

for turn in conv.get('turns', []):
    if 'text' not in turn:
        turn['text'] = turn.get('content', turn.get('message', ''))

result = score_conversation_synthetic(conv, facets)
result['scoring_mode'] = 'synthetic_fallback'

with open('data/conversations/scored/conv_005_scored.json', 'w') as f:
    json.dump(result, f, indent=2)

print('conv_005 fixed, score=', result['overall_score'])