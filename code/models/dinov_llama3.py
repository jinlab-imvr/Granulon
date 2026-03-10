import os
import torch
from PIL import Image
from transformers import AutoConfig,AutoTokenizer,AddedToken,AutoImageProcessor,AutoModelForCausalLM,AutoModel
from transformers import LlavaProcessor,LlavaConfig,LlavaForConditionalGeneration
from transformers import DINOv3ViTConfig
from custom.custom_llava import CustomLlavaForConditionalGeneration, CustomLlavaProcessor



llm_model_name = 'meta-llama/Llama-3.2-3B-Instruct'
# llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
vision_model_name = 'facebook/dinov3-vitl16-pretrain-lvd1689m'

print("llm_model_name:",llm_model_name)
print("vision_model_name:",vision_model_name)

tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True, use_fast=False)

tokenizer.add_tokens([AddedToken("<image>", special=True, normalized=False)], special_tokens=True)
image_token_index = tokenizer.encode("<image>", add_special_tokens=False)

assert len(image_token_index) == 1, "Tokenizer failed to register <image> as a single token!"
print("image_token_index:",image_token_index)
print(len(image_token_index))
image_token_index = image_token_index[0]


image_processor = AutoImageProcessor.from_pretrained(vision_model_name)

processor = CustomLlavaProcessor(tokenizer=tokenizer, image_processor=image_processor,
                            patch_size =16,  # ✅ DINOv3 是 patch16
                            vision_feature_select_strategy="full",
                            image_token="<image>",
                            num_additional_image_tokens=0)

print("image_token_index:",image_token_index)


full_vision = AutoModel.from_pretrained(vision_model_name)
vision_model = getattr(full_vision, "vision_model", full_vision) 
vision_model.eval()
with torch.no_grad():
    dummy = Image.new("RGB", (224, 224), color=0)  
    pv = image_processor(images=dummy, return_tensors="pt")["pixel_values"]
    out = vision_model(pv)
    seq_len = out.last_hidden_state.shape[1]
    hidden_size = out.last_hidden_state.shape[2]

print("detected vision seq_len:", seq_len)
print("detected hidden_size:", hidden_size)


llm_config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)


vision_config = DINOv3ViTConfig.from_pretrained(vision_model_name)
vision_config.vision_use_head = False
config = LlavaConfig(
        text_config=llm_config,
        vision_config=vision_config,
        image_token_index=image_token_index,
        vision_feature_layer=-1,
        vision_feature_select_strategy="full",  
        image_seq_length=seq_len - 1,
)


# model = LlavaForConditionalGeneration(config)
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
# raise SystemExit("stop here for debug")

model.language_model = AutoModelForCausalLM.from_pretrained(llm_model_name, trust_remote_code=True)



if hasattr(model, "config"):
    model.config.use_cache = False
if hasattr(model.language_model, "config"):
    model.language_model.config.use_cache = False

dino_full = AutoModel.from_pretrained(vision_model_name)
missing = model.vision_tower.load_state_dict(dino_full.state_dict(), strict=False)



export_path = '/dinov_llama3'
os.makedirs(export_path, exist_ok=True)
tokenizer.save_pretrained(export_path)
model.config.save_pretrained(export_path)
processor.save_pretrained(export_path)
model.save_pretrained(export_path)
print("model save done:",export_path)

