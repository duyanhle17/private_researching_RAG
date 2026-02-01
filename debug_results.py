# debug_results.py
import json

with open('enhanced_results.json') as f:
    data = json.load(f)

# Tìm các câu có context rỗng hoặc graph_score = 0 cho tất cả
no_kg_facts = []
all_zero_graph = []

for i, item in enumerate(data):
    kg_facts = item.get('kg_facts', [])
    scores = item.get('retrieval_scores', [])
    
    # Check if all graph scores are 0
    if scores and all(s['graph'] == 0.0 for s in scores):
        all_zero_graph.append({
            'idx': i+1,
            'question': item['question'][:80],
            'kg_facts_count': len(kg_facts)
        })
    
    if not kg_facts:
        no_kg_facts.append(i+1)

print(f'📊 Total questions: {len(data)}')
print(f'📊 Questions with 0 KG facts: {len(no_kg_facts)}')
print(f'📊 Questions with all graph_score=0: {len(all_zero_graph)}')
print()
print('=== Questions with all graph_score=0 ===')
for item in all_zero_graph[:20]:
    print(f"  [{item['idx']}] kg_facts={item['kg_facts_count']} | {item['question']}")

print()
print('=== Sample questions with KG facts (good) ===')
for i, item in enumerate(data[:30]):
    if item.get('kg_facts'):
        print(f"  [{i+1}] kg_facts={len(item['kg_facts'])} | {item['question'][:60]}...")
