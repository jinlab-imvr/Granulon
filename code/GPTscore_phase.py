import json, os, base64, tqdm, re
from openai import OpenAI
import numpy as np

# ========== 1. 配置 ==========
client = OpenAI(
    base_url="",
    api_key=""
)

IMAGE_ROOT = "//images"
MODEL_JSON = "/eval.json"
SAVE_FILE  = "./reason/dinov_llama3.jsonl"
# ======================================


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def split_into_segments(text: str, num_segments: int = 5):

    sentences = re.split(r'(?<=[.!?。！？])\s*', text.strip())
    n = len(sentences)
    if n == 0:
        return [""] * num_segments
    seg_size = max(1, n // num_segments)
    segments = []
    for i in range(num_segments):
        start = i * seg_size
        end = (i + 1) * seg_size if i < num_segments - 1 else n
        seg_text = " ".join(sentences[start:end]).strip()
        segments.append(seg_text if seg_text else "(empty)")
    return segments


def build_prompt(pred_segments):
    seg_text = "\n\n".join([f"Segment {i+1}: {s}" for i, s in enumerate(pred_segments)])
    return f"""
You are an expert evaluator for AI-generated image descriptions.
Given the image and five text segments produced by a model, evaluate **each segment independently**.

For every segment, provide:
1. **ACCURACY SCORE (0–100)** — How well this segment matches the actual content of the image.
2. **HALLUCINATION SCORE (0–100)** — How much content in this segment is NOT present in the image.

Return ONLY a valid JSON object in this format:
{{
  "accuracy_scores": {{
    "segment_1": int,
    "segment_2": int,
    "segment_3": int,
    "segment_4": int,
    "segment_5": int
  }},
  "hallucination_scores": {{
    "segment_1": int,
    "segment_2": int,
    "segment_3": int,
    "segment_4": int,
    "segment_5": int
  }}
}}

MODEL OUTPUT (split into 5 chronological segments):
{seg_text}
"""


def call_gpt4o(image_b64: str, prompt: str):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.1
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print("GPT-4o error:", e)
        return None


def main():
    with open(MODEL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    fw = open(SAVE_FILE, "w", encoding="utf-8")

    acc_sums = np.zeros(5)
    hal_sums = np.zeros(5)
    valid_cnt = 0

    for idx, item in enumerate(tqdm.tqdm(data, desc="Evaluating")):
        img_path = item["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(IMAGE_ROOT, os.path.basename(img_path))

        b64 = encode_image(img_path)
        pred = item["prediction"]
        segments = split_into_segments(pred, 5)

        prompt = build_prompt(segments)
        result = call_gpt4o(b64, prompt)

        if result is None:
            result = {
                "accuracy_scores": {f"segment_{i+1}": -1 for i in range(5)},
                "hallucination_scores": {f"segment_{i+1}": -1 for i in range(5)}
            }

        # Write single-line result
        out = {"sample_id": idx, "image": img_path, "scores": result}
        fw.write(json.dumps(out, ensure_ascii=False) + "\n")

        # Accumulate valid scores
        acc_valid = all(v != -1 for v in result["accuracy_scores"].values())
        if acc_valid:
            for i in range(5):
                acc_sums[i] += result["accuracy_scores"][f"segment_{i+1}"]
                hal_sums[i] += result["hallucination_scores"][f"segment_{i+1}"]
            valid_cnt += 1

    fw.close()

    # ===== Output average =====
    if valid_cnt == 0:
        print("done, but no valid samples.")
        return

    avg_acc = acc_sums / valid_cnt
    avg_hal = hal_sums / valid_cnt

    print("\n========== Average Scores (by Segment) ==========")
    for i in range(5):
        print(f"Segment {i+1}: Accuracy = {avg_acc[i]:.2f} / 100, "
              f"Hallucination = {avg_hal[i]:.2f} / 100")
    print(f"Valid samples / Total samples = {valid_cnt} / {len(data)}")
    print("Individual results saved to:", SAVE_FILE)


if __name__ == "__main__":
    main()
