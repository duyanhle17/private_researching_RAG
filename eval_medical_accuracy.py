import os
import json
import logging
import re
import httpx
from openai import OpenAI
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_client():
    api_key = os.getenv("NVAPI_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing NVAPI_KEY in environment.")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        http_client=httpx.Client(timeout=60.0),
    )

def evaluate_answer(question, answer, groundtruth, client):
    prompt = f"""You are a professional medical evaluator. Compare the 'Model Answer' against the 'Ground Truth' for accuracy.

Question: {question}
Ground Truth: {groundtruth}
Model Answer: {answer}

Evaluation Criteria:
1.0 (Correct): The model answer captures the core medical facts present in the ground truth.
0.7 (Partial): The model answer is partially correct or contains some relevant info but misses key specifics.
0.0 (Wrong): The model answer is incorrect, irrelevant, or says 'not stated' when the info exists.

Provide a very brief reasoning, then output the score in this format:
Score: X.X
"""
    try:
        response = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        content = response.choices[0].message.content
        match = re.search(r"Score:\s*([01]\.[05])", content)
        if match:
            return float(match.group(1)), content
        return 0.0, content
    except Exception as e:
        return 0.0, f"Error: {str(e)}"

def main():
    prediction_file = "sat_baseline_v2_medical_predictions.json"
    if not os.path.exists(prediction_file):
        print(f"File {prediction_file} not found.")
        return

    with open(prediction_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = get_client()
    scores = []
    results = []

    print(f"Evaluating {len(data)} questions...")
    for item in tqdm(data):
        score, reasoning = evaluate_answer(item['question'], item['answer'], item['groundtruth'], client)
        scores.append(score)
        results.append({
            "question": item['question'],
            "score": score,
            "reasoning": reasoning.strip()
        })

    avg_score = sum(scores) / len(scores) if scores else 0
    correct_count = sum(1 for s in scores if s == 1.0)
    partial_count = sum(1 for s in scores if s == 0.5)
    wrong_count = sum(1 for s in scores if s == 0.0)

    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"Total Questions: {len(data)}")
    print(f"Correct (1.0):   {correct_count}")
    print(f"Partial (0.5):   {partial_count}")
    print(f"Wrong   (0.0):   {wrong_count}")
    print(f"Average Accuracy: {avg_score*100:.2f}%")
    print("="*50)

    # Save detailed evaluation
    with open("detailed_accuracy_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": len(data),
                "correct": correct_count,
                "partial": partial_count,
                "wrong": wrong_count,
                "average_score": avg_score
            },
            "details": results
        }, f, indent=2, ensure_ascii=False)
    
    print("Detailed report saved to: detailed_accuracy_report.json")

if __name__ == "__main__":
    main()
