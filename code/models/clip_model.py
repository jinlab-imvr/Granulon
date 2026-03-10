import os
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AddedToken,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModel,
)
from transformers import (
    LlavaProcessor,
    LlavaConfig,
    LlavaForConditionalGeneration,
)
from transformers import CLIPVisionConfig
from custom.custom_clip import CustomLlavaForConditionalGeneration

llm_model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
vision_model_name = 'openai/clip-vit-large-patch14'

print("llm_model_name:", llm_model_name)
print("vision_model_name:", vision_model_name)

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)

image_token_index = tokenizer.encode('<image>', add_special_tokens=False)[0]
print("image_token_index:", image_token_index)

image_processor = AutoImageProcessor.from_pretrained(vision_model_name)

processor = LlavaProcessor(
    tokenizer=tokenizer,
    image_processor=image_processor,
    patch_size=14,
    vision_feature_select_strategy="default",
    image_token="<image>",
    num_additional_image_tokens=1,
)
print("image_token_index:", image_token_index)

with torch.no_grad():
    dummy = Image.new("RGB", (224, 224), color=0)
    pv = image_processor(images=dummy, return_tensors="pt")["pixel_values"]
    vision_backbone = AutoModel.from_pretrained(vision_model_name)
    vision_backbone.eval()
    vision_model_tmp = vision_backbone.vision_model  # CLIPVisionModel
    out = vision_model_tmp(pv)
    seq_len_total = out.last_hidden_state.shape[1]
    image_seq_length = seq_len_total - 1
print(f"Detected CLIP seq_len_total={seq_len_total}, image_seq_length(without CLS)={image_seq_length}")

llm_config = AutoConfig.from_pretrained(llm_model_name)

vision_config = CLIPVisionConfig.from_pretrained(vision_model_name)

config = LlavaConfig(
    text_config=llm_config,
    vision_config=vision_config,
    image_token_index=image_token_index,
    vision_feature_layer=-1,  
    image_seq_length=image_seq_length, 
)

model = CustomLlavaForConditionalGeneration(config)
print("model init done")

print("\n====== Projector Structure ======")
if hasattr(model, "multi_modal_projector"):
    print(model.multi_modal_projector)
elif hasattr(model, "projector"):
    print(model.projector)
else:
    print("⚠️ No projector found in model (check your LlavaConfig or model structure).")
print("=================================\n")

model.language_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
vocab_size = len(tokenizer)
print(f">>> Tokenizer size: {vocab_size}")

vision_model = AutoModel.from_pretrained(vision_model_name).vision_model  # CLIPVisionModel

statea = model.vision_tower.vision_model.state_dict().keys()
stateb = vision_model.state_dict().keys()
try:
    assert len(statea - stateb) == 0
except AssertionError:
    print("Warning: some keys differ between target and source vision models.")
    print("Missing in source:", list(statea - stateb)[:10])
    print("Unexpected in source:", list(stateb - statea)[:10])

incomp = model.vision_tower.vision_model.load_state_dict(vision_model.state_dict(), strict=False)
print("load_state_dict missing_keys:", getattr(incomp, "missing_keys", []))
print("load_state_dict unexpected_keys:", getattr(incomp, "unexpected_keys", []))

print("model load done")

export_path = './clip_qwen'
tokenizer.save_pretrained(export_path)
model.config.save_pretrained(export_path)
processor.save_pretrained(export_path)
model.save_pretrained(export_path)
print("model save done:", export_path)