# check_results.py
import json
import re

def normalize(text):
    """Normalize text for comparison"""
    text = text.lower().strip()
    # Remove common prefixes
    for prefix in ["it is ", "it's ", "it states ", "it says ", "they are ", "he is ", "she is ", "ucf is ", "the "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    # Remove punctuation at end
    text = re.sub(r'[.,!?]+$', '', text)
    return text

def fuzzy_match(answer, groundtruth):
    """Check if answer and groundtruth are semantically similar"""
    ans_norm = normalize(answer)
    gt_norm = normalize(groundtruth)
    
    # Exact or substring match after normalization
    if ans_norm == gt_norm or ans_norm in gt_norm or gt_norm in ans_norm:
        return True
    
    # Key phrase overlap
    ans_words = set(ans_norm.split())
    gt_words = set(gt_norm.split())
    overlap = len(ans_words & gt_words) / max(len(gt_words), 1)
    
    return overlap >= 0.6  # At least 60% word overlap

with open('enhanced_results.json') as f:
    data = json.load(f)

print(f"📊 Total: {len(data)} questions")
print()

# Strict accuracy
strict_correct = sum(1 for o in data if o["groundtruth"].lower() in o["answer"].lower())
print(f"📏 Strict match (GT in Answer): {strict_correct}/{len(data)} ({100*strict_correct/len(data):.1f}%)")

# Fuzzy accuracy  
fuzzy_correct = sum(1 for o in data if fuzzy_match(o["answer"], o["groundtruth"]))
print(f"🔄 Fuzzy match (60% overlap): {fuzzy_correct}/{len(data)} ({100*fuzzy_correct/len(data):.1f}%)")

# Count "not stated in the text"
not_found = sum(1 for o in data if "not stated" in o["answer"].lower())
print(f"❌ 'Not stated in text': {not_found}/{len(data)}")

print()
print("=== Sample Results ===")
for i, item in enumerate(data[:15], 1):
    q = item['question'][:55]
    a = item['answer'][:55]
    g = item['groundtruth'][:55]
    strict = "✅" if item['groundtruth'].lower() in item['answer'].lower() else "❌"
    fuzzy = "🔄" if fuzzy_match(item['answer'], item['groundtruth']) else "  "
    print(f"\n[{i}] {strict}{fuzzy}")
    print(f"   Q: {q}...")
    print(f"   A: {a}...")
    print(f"   GT: {g}...")
