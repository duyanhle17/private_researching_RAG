import json

v1 = json.load(open('sat_baseline_results.json', encoding='utf-8'))
v2 = json.load(open('sat_baseline_v2_entities_results.json', encoding='utf-8'))

total = len(v2)

# Substring accuracy
c1 = sum(1 for o in v1 if o.get('groundtruth','').lower().strip() in o.get('answer','').lower())
c2 = sum(1 for o in v2 if o.get('groundtruth','').lower().strip() in o.get('answer','').lower())

# Retrieval stats v2
entity_found  = sum(1 for o in v2 if o['matched_entities'])
phrase_found  = sum(1 for o in v2 if o['retrieval'].get('phrase_chunks', 0) > 0)
bm25_only     = sum(1 for o in v2 if not o['matched_entities'] and o['retrieval']['bm25_chunks'] > 0)
avg_entity    = sum(o['retrieval']['entity_chunks'] for o in v2) / total
avg_phrase    = sum(o['retrieval'].get('phrase_chunks', 0) for o in v2) / total
avg_bm25      = sum(o['retrieval']['bm25_chunks'] for o in v2) / total
avg_semantic  = sum(o['retrieval']['semantic_chunks'] for o in v2) / total
avg_total     = sum(o['retrieval']['total_chunks'] for o in v2) / total

print(f"=== ACCURACY ===")
print(f"v1 (Semantic+Graph Rerank):    {c1}/{len(v1)} = {100*c1/len(v1):.1f}%")
print(f"v2 (Entity+Phrase+BM25+FAISS): {c2}/{total} = {100*c2/total:.1f}%")

print(f"\n=== v2 RETRIEVAL STATS ===")
print(f"Total questions:   {total}")
print(f"Entity matched:    {entity_found}/{total} ({100*entity_found/total:.1f}%)")
print(f"Phrase matched:    {phrase_found}/{total} ({100*phrase_found/total:.1f}%)")
print(f"BM25-only (no ent):{bm25_only}/{total} ({100*bm25_only/total:.1f}%)")
print(f"Avg entity chunks: {avg_entity:.2f}")
print(f"Avg phrase chunks: {avg_phrase:.2f}")
print(f"Avg BM25 chunks:   {avg_bm25:.2f}")
print(f"Avg semantic chks: {avg_semantic:.2f}")
print(f"Avg total chunks:  {avg_total:.2f}")

print(f"\n=== PER-QUESTION BREAKDOWN ===")
for i, o in enumerate(v2, 1):
    r = o['retrieval']
    ph = r.get('phrase_chunks', 0)
    sub = o.get('groundtruth','').lower().strip() in o.get('answer','').lower()
    flag = "✅" if sub else "  "
    ents = ','.join(o['matched_entities'][:2]) if o['matched_entities'] else '-'
    print(f"{flag}[{i:02d}] E={r['entity_chunks']} Ph={ph} B={r['bm25_chunks']} S={r['semantic_chunks']} | {ents[:25]:<25} | {o['answer'][:55]}")
