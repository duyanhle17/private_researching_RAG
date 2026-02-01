import pickle
import json
import random
from tqdm import tqdm

KG_PATH = "sat_data/kg.pkl"
OUT_PATH = "sat_data/sat_lp_train.jsonl"

NUM_SAMPLES = 5000        # có thể giảm xuống 2000 nếu KG nhỏ
CONTEXT_SIZE = 4          # số fact cho mỗi graph

def extract_triples(kg):
    triples = []
    for h, t, data in kg.edges(data=True):
        r = data.get("relation", "related_to")
        triples.append((h, r, t))
    return triples


def build_sat_lp_dataset(triples):
    data = []

    for _ in tqdm(range(NUM_SAMPLES)):
        h, r, t = random.choice(triples)

        context = random.sample(triples, min(CONTEXT_SIZE, len(triples)))
        graph_lines = []

        for x, rel, y in context:
            if (x, rel, y) == (h, r, t):
                graph_lines.append(f"{x} {rel} ?")
            else:
                graph_lines.append(f"{x} {rel} {y}")

        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a graph-aware reasoning model."
                },
                {
                    "role": "user",
                    "content":
                        "<GRAPH>\n"
                        + "\n".join(graph_lines)
                        + "\n</GRAPH>\n\n"
                        + "Task: Predict the missing entity."
                },
                {
                    "role": "assistant",
                    "content": t
                }
            ]
        }

        data.append(sample)

    return data


if __name__ == "__main__":
    print("📦 Loading KG...")
    with open(KG_PATH, "rb") as f:
        kg = pickle.load(f)

    triples = extract_triples(kg)

    print(f"✅ Loaded KG with {len(triples)} triples")

    if len(triples) < 50:
        raise ValueError("❌ KG quá nhỏ để train SAT")

    dataset = build_sat_lp_dataset(triples)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in dataset:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"🎉 SAT dataset saved to {OUT_PATH}")
