"""Debug entity extraction for failing questions."""
import sys

with open('data/medical/id2title.txt') as f:
    title2eid = {}
    id2title = {}
    for line in f:
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            eid = int(parts[0])
            title = parts[1]
            id2title[eid] = title
            if len(title) > 2:
                title2eid[title.lower()] = eid

sorted_titles = sorted(title2eid.keys(), key=len, reverse=True)

def extract_entities(question):
    question_lower = question.lower()
    matched = []
    used_spans = []
    for title_lower in sorted_titles:
        pos = question_lower.find(title_lower)
        if pos == -1:
            continue
        end = pos + len(title_lower)
        overlap = any(pos < e and end > s for s, e in used_spans)
        if not overlap:
            eid = title2eid[title_lower]
            matched.append((eid, id2title.get(eid, title_lower)))
            used_spans.append((pos, end))
    return matched

# Test on failing questions
questions = [
    'Which diagnostic methods are used for BCC?',
    'What does follow-up for BCC typically include?',
    'Is sun exposure a risk factor for BCC?',
    'Do tanning beds increase the risk of BCC?',
    'Which systemic therapy may be considered for BCC?',
    'Is autoimmune disease associated with increased BCC risk?',
    'Does older age influence the risk of BCC?',
    'What is recommended for follow-up after BCC treatment?',
    'What is included in annual follow-up for BCC patients?',
    'Which exams are essential for diagnosing BCC?',
]

print("=== ENTITY EXTRACTION ON FAILING QUESTIONS ===\n")
for q in questions:
    matches = extract_entities(q)
    names = [name for _, name in matches]
    print(f'Q: {q}')
    print(f'  Matched: {names}')
    print()

# Check some specific substring match issues
print("\n=== SUBSTRING MATCH ANALYSIS ===\n")

# What does "treatment" match?
test_words = ['treatment', 'follow-up', 'diagnostic', 'tanning beds', 'BCC', 'annual', 'diagnosing']
for word in test_words:
    word_lower = word.lower()
    matches = []
    for title_lower, eid in title2eid.items():
        if title_lower in word_lower or word_lower in title_lower:
            matches.append((eid, id2title[eid], title_lower))
    if matches:
        print(f'Word "{word}" overlaps with:')
        for eid, title, tl in matches[:5]:
            print(f'  ID={eid}: "{title}"')
    else:
        print(f'Word "{word}": NO match')
    print()

# Count how many titles are substrings of common English words
print("\n=== NOISY SHORT ENTITIES (len 3-4) ===\n")
noisy = [(eid, t) for t, eid in title2eid.items() if 3 <= len(t) <= 4]
noisy.sort(key=lambda x: x[1])
print(f"Total entities with title length 3-4: {len(noisy)}")
for eid, title in noisy[:50]:  # Show first 50
    full_title = id2title[eid]
    print(f'  ID={eid}: "{full_title}"')
