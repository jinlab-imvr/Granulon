import argparse
import json
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
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
import numpy as np
from scipy.spatial.distance import cosine

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
            output_hidden_states=True
        )
        outputs = model(**inputs, output_hidden_states=True)

    hs = outputs.hidden_states  
    if hs is None:
        raise RuntimeError("Model did not return hidden states.")

    prompt_len = inputs["input_ids"].shape[1]
    gen_only = out_ids[0][prompt_len:]
    return processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_text_image_hidden(model, processor: LlavaProcessor, text: str, image: Image.Image, max_new_tokens: int) -> str:
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
        outputs = model(**inputs, output_hidden_states=True)

    hs = outputs.hidden_states 
    if hs is None:
        raise RuntimeError("Model did not return hidden states.")

    return hs

@torch.inference_mode()
def generate_text_only_hidden(model, processor, text: str, max_new_tokens: int):
    messages = [{"role": "user", "content": text}]
    try:
        apply_fn = getattr(processor, "apply_chat_template", None)
        if apply_fn is not None:
            prompt = apply_fn(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        prompt = text

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor has no tokenizer; cannot run text-only generation.")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hs = outputs.hidden_states  
    if hs is None:
        raise RuntimeError("Model did not return hidden states.")
   
    return hs


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
    eval_dir = Path("./eval")
    # images_dir = eval_dir.joinpath("images")
    # eval_dir.mkdir(parents=True, exist_ok=True)
    # images_dir.mkdir(parents=True, exist_ok=True)
    similarity_matrix = np.zeros((len(samples), 28)) 
    for i, ex in enumerate(samples, 1):
        print(f"\n===== Sample {i}/{len(samples)} =====")
        img = Image.open(ex['image_path']).convert("RGB")
        img = img.resize((224, 224))
        pred = gen_fn(ex["human_question"], img)
        predictions.append(pred)
        references.append(ex["reference"])
        hidden_layer_pred = generate_text_image_hidden(model, proc, ex["human_question"], img, args.max_new_tokens)
        hidden_layer_ref = generate_text_only_hidden(model, proc, ex["reference"], args.max_new_tokens)
        for layer_idx in range(28):  
            sample_layer = hidden_layer_pred[layer_idx + 1]
            ref_layer = hidden_layer_ref[layer_idx + 1]
            sample_layer_mean = torch.mean(sample_layer, dim=1)  
            ref_layer_mean = torch.mean(ref_layer, dim=1)       
            sample_vec = sample_layer_mean.detach().cpu().to(torch.float32).numpy().squeeze() 
            ref_vec = ref_layer_mean.detach().cpu().to(torch.float32).numpy().squeeze()       
            similarity = 1 - cosine(sample_vec, ref_vec)
            similarity_matrix[i - 1, layer_idx] = similarity 
            print(f"Layer {layer_idx + 1} Similarity: {similarity:.4f}")

    plt.figure(figsize=(12, 8))
    h = sns.heatmap(similarity_matrix, 
                cmap="viridis",
                annot=False,
                fmt=".2f",
                xticklabels=[f"{k+1}" for k in range(28)],
                yticklabels=[f"{k+1}" for k in range(10)],
                cbar_kws={"label": "Value"})
    cbar = h.collections[0].colorbar               
    cbar.set_label("Value", fontsize=26)          
    cbar.ax.tick_params(labelsize=20)            
    plt.title("Layer-wise State Alignment Heatmap (CLIP)", fontsize=26)
    plt.xlabel("Layers", fontsize=26)
    plt.ylabel("Samples", fontsize=26)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    heatmap_path = eval_dir.joinpath("similarity_heatmap_CLIP.png")
    plt.savefig(heatmap_path, dpi=300) 
    print(f"Heatmap saved to: {heatmap_path}")
    plt.show()


if __name__ == "__main__":
    main()