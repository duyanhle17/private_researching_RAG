# debug_entity_extraction.py
"""Debug tại sao một số queries không match được entities"""
import json
import pickle
import spacy

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Load chunk_entities
with open('enhanced_sat_data/chunk_entities.pkl', 'rb') as f:
    chunk_entities = pickle.load(f)

# Load entity2id
with open('enhanced_sat_data/entity2id.pkl', 'rb') as f:
    entity2id = pickle.load(f)

# Test queries
test_queries = [
    "What is a government, according to the given definition?",
    "Who gives the Satellite Awards, according to the text?",
    "Where is Tottenham Hotspur based?",
    "How were The Killers formed and in what year?",
    "What institutional type is the University of Central Florida (UCF)?",  # This one works
    "Where is UCF's main campus located?",  # This one works
]

print("=" * 60)
print("DEBUG: Entity Extraction from Queries")
print("=" * 60)

for q in test_queries:
    doc = nlp(q)
    
    # spaCy NER
    ner_ents = [ent.text.strip().lower() for ent in doc.ents]
    
    # Check if any match in entity2id
    matches = [e for e in ner_ents if e in entity2id]
    
    print(f"\n📝 Query: {q[:70]}...")
    print(f"   NER entities: {ner_ents}")
    print(f"   Matched in KG: {matches}")

print("\n" + "=" * 60)
print("DEBUG: Sample entities in KG")
print("=" * 60)
sample_entities = list(entity2id.keys())[:30]
print(f"First 30 entities: {sample_entities}")

print("\n" + "=" * 60)
print("DEBUG: Search for specific entities")
print("=" * 60)
search_terms = ['government', 'tottenham', 'satellite', 'killers', 'ucf', 'university', 'central florida']
for term in search_terms:
    found = [e for e in entity2id.keys() if term in e.lower()]
    print(f"'{term}': {found[:5]}")
