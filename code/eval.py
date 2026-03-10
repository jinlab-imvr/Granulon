import argparse
import json
import os
import sys
from typing import List, Dict, Any
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, LlavaProcessor, LlavaForConditionalGeneration
import base64
import io
import shutil
from peft import PeftModel, PeftConfig
from pycocoevalcap.bleu.bleu import Bleu
try:
    from transformers import AutoModelForImageTextToText
    HAS_IT2T = True
except Exception:
    HAS_IT2T = False

from models.custom.custom_clip import CustomLlavaForConditionalGeneration as ClipLlavaForConditionalGeneration

def parse_samples(index_path: str, dataset_dir: str, num_samples: int) -> List[Dict[str, Any]]:
    samples = []
    index_path = Path(index_path)
    dataset_dir = Path(dataset_dir)
    chat_data = pd.read_json(index_path).to_dict(orient="records")
    image_dir = dataset_dir.joinpath("images") 
    for index in range(num_samples):
        cur_data = chat_data[index+500]  # {'id': 'GCC_train_002109690', 'image': 'GCC_train_002109690.jpg', 'conversations': [{...}, {...}]}
        conversations = cur_data.get("conversations")  # [{'from': 'human', 'value': 'Offer a succinct explanation of the picture presented.\n<image>'}, {'from': 'gpt', 'value': "it 's appropriate for teens to want to spend more time with their peers than their parents as they get older ."}]

        human_input = conversations[0].get("value")   # 'Offer a succinct explanation of the picture presented.\n<image>'
        chatbot_output = conversations[1].get("value")  # "it 's appropriate for teens to want to spend more time with their peers than their parents as they get older ."

        image_path = image_dir.joinpath(cur_data.get("image"))
        samples.append({
            "image_path": str(image_path),
            "human_question": human_input, 
            "reference": chatbot_output
        })
    return samples


@torch.inference_mode()
def generate_qwen(model, processor, text: str, image: Image.Image, max_new_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()} 
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        pad_token_id=getattr(processor.tokenizer, "eos_token_id", None),
    )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_llava(model, processor: LlavaProcessor, text: str, image: Image.Image, max_new_tokens: int) -> str:
    if "<image>" not in text:
        raise ValueError("<image> not in question for LLaVA input.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(image, prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_only = out_ids[0][prompt_len:]
    return processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()



def load_qwen_model(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = dict(trust_remote_code=True, dtype=dtype)
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"  
    else:
        model_kwargs["low_cpu_mem_usage"] = True 

    if HAS_IT2T:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        print("Using AutoModelForImageTextToText.")
    else:
        sys.exit("ERROR: Neither AutoModelForImageTextToText nor AutoModelForVision2Seq is available in this transformers version.")
    model.eval()
    return model, processor


def load_llava_model(model_path: str, use_custom: str = "None", use_lora: str = "NO"):
    """
    Load a LLaVA model according to use_custom:
      - "All": use both CustomLlavaProcessor and AllLlavaForConditionalGeneration (requires HAS_CUSTOM_LLAVA_FULL)
      - "Part": use standard LlavaProcessor but PartLlavaForConditionalGeneration as the model (requires HAS_CUSTOM_LLAVA_PART)
      - "None": use standard LlavaProcessor and LlavaForConditionalGeneration from transformers
    """
    model_path = Path(model_path)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Auto device placement when CUDA is available
    model_kwargs = dict(trust_remote_code=True, dtype=dtype)
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["low_cpu_mem_usage"] = True

    if use_custom == "clip":
        # Use standard LlavaProcessor but the CLIP custom model class
        processor = LlavaProcessor.from_pretrained(model_path, local_files_only=True)
        print("Loading custom CLIP LLaVA model from:", model_path)
        model = ClipLlavaForConditionalGeneration.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )

    model.eval()
    return model, processor


from bert_score import score

def bertscore_metric(predictions: list[str], references: list[str]) -> float:

    P, R, F1 = score(predictions, references, lang="en", verbose=False)
    return float(F1.mean().item())

def recall_metric(predictions: list[str], references: list[str]) -> float:

    import re
    
    recalls = []
    for pred, ref in zip(predictions, references):
        pred_tokens = re.findall(r"\w+", pred.lower())
        ref_tokens = re.findall(r"\w+", ref.lower())
        if not ref_tokens:
            continue
        overlap = len(set(pred_tokens) & set(ref_tokens))
        recalls.append(overlap / len(set(ref_tokens)))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def _coco_format(predictions: List[str], references: List[str]):
    preds = {i: [p] for i, p in enumerate(predictions)}
    gts   = {i: [r] for i, r in enumerate(references)}
    return preds, gts


def metrics_all(predictions: List[str], references: List[str]) -> Dict[str, float]:
    preds, gts = _coco_format(predictions, references)

    bleu = Bleu(4)
    bleu_scores, _ = bleu.compute_score(gts, preds)

    bert_f1 = bertscore_metric(predictions, references)
    recall   = recall_metric(predictions, references)

    return {
        "BLEU-1": bleu_scores[0],
        "BLEU-2": bleu_scores[1],
        "BLEU-3": bleu_scores[2],
        "BLEU-4": bleu_scores[3],
        "BERTScore-F1": bert_f1,
        "Recall": recall,
    }

def main():
    parser = argparse.ArgumentParser(description="Minimal eval on FLUX-Reason.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["qwen2p5_vl", "llava"], required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--use_lora", type=str, choices=["YES", "NO"], default="NO",
                        help="Whether to use LoRA adapter when loading LLaVA model.")
    parser.add_argument("--use_custom_model", type=str, choices=["clip"], default="None",
                        help="Whether to use custom LLaVA components: All=processor+model, Part=model only, None=standard")
    args = parser.parse_args()
    index_path = Path(args.dataset_dir)
    index_path = index_path.joinpath("chat.json")
    print(f"Using index file: {index_path}")
    # print(f'type of samples: {type(samples)}    number of samples: {len(samples)}')
    # print(f'type of samples[0]: {type(samples[0])}    keys of samples[0]: {list(samples[0].keys())}')
    samples = parse_samples(index_path, args.dataset_dir, args.num_samples)

    if args.model_type == "qwen2p5_vl":
        model, proc = load_qwen_model(args.model)
        gen_fn = lambda text, img: generate_qwen(model, proc, text, img, args.max_new_tokens)
    else:
        model, proc = load_llava_model(args.model, args.use_custom_model, args.use_lora)
        gen_fn = lambda text, img: generate_llava(model, proc, text, img, args.max_new_tokens)
    
    samples_out = []
    predictions = []
    references = []

    model_name = Path(args.model).name
    eval_dir = Path("./eval/reason/" + f"{model_name}" )
    images_dir = eval_dir.joinpath("images")
    eval_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, ex in enumerate(samples, 1): 
        print(f"\n===== sample {i}/{len(samples)} =====")
        print(f"human_question: {ex['human_question']}")
        print(f"reference: {ex['reference']}")

        # image_bytes = base64.b64decode(ex['image'])
        # img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = Image.open(ex['image_path']).convert("RGB")
        img = img.resize((224, 224))
        pred = gen_fn(ex["human_question"], img)
        predictions.append(pred)
        references.append(ex["reference"])

        print(f"pred: {pred}")

        src_img = Path(ex["image_path"])
        if not src_img.exists():
            print(f"Warning: image not found, skipping: {src_img}")
            continue
        dst_img = images_dir.joinpath(src_img.name)
        shutil.copy2(src_img, dst_img)
        samples_out.append({
            "image_path": str(dst_img), 
            "human_question": ex["human_question"],
            "reference": ex["reference"],
            "prediction": pred
        })

    json_path = eval_dir.joinpath("eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples_out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(samples_out)} samples to {json_path}")

    # accuracy = compute_accuracy(predictions, references)
    accuracy = metrics_all(predictions, references)
    print(f"\n===== result =====")
    # print(f"MODEL={args.model} TYPE={args.model_type} ACCURACY={accuracy:.6f} TOTAL={len(samples)}")
    # print(f"MODEL={args.model} TYPE={args.model_type} Score={accuracy['bertscore_f1']:.6f}, Recall={accuracy['recall']:.4f} TOTAL={len(samples)}")
    print(f"MODEL={args.model} TYPE={args.model_type} Score={accuracy} TOTAL={len(samples)}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()