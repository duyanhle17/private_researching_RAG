# analyze_scores.py
import json

with open('enhanced_results.json') as f:
    data = json.load(f)

print('=== SCORE ANALYSIS ===')
print()

# Analyze scores
low_semantic = []
low_combined = []

for i, item in enumerate(data):
    scores = item.get('retrieval_scores', [])
    if scores:
        max_sem = max(s['semantic'] for s in scores)
        max_comb = max(s['combined'] for s in scores)
        
        if max_sem < 0.3:
            low_semantic.append((i+1, max_sem, item['question'][:50]))
        if max_comb < 0.3:
            low_combined.append((i+1, max_comb, item['question'][:50]))

print(f'Questions with max semantic score < 0.3: {len(low_semantic)}')
print(f'Questions with max combined score < 0.3: {len(low_combined)}')
print()

print('=== Top 10 Low Scoring Questions ===')
for idx, sem, q in sorted(low_semantic, key=lambda x: x[1])[:10]:
    print(f'  [{idx}] sem={sem:.3f} | {q}...')

print()
print('=== Sample High vs Low Scoring ===')
high = [item for item in data if item.get('retrieval_scores') and max(s['semantic'] for s in item['retrieval_scores']) > 0.5]
low = [item for item in data if item.get('retrieval_scores') and max(s['semantic'] for s in item['retrieval_scores']) < 0.2]

print(f'High scoring (>0.5): {len(high)} questions')
print(f'Low scoring (<0.2): {len(low)} questions')
