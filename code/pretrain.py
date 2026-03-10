import logging
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
)
from callback import ProjectorGradMonitor 
from custom.custom_llava import CustomLlavaForConditionalGeneration, CustomLlavaProcessor

from data import LlavaDataset, TrainLLavaModelCollator
from util import print_trainable_parameters
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)



class LossLoggingCallback(transformers.TrainerCallback):

    def __init__(self):
        self.logger = logging.getLogger("training.loss")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = getattr(state, "global_step", None)
        for k, v in (logs.items() if isinstance(logs, dict) else []):
            if isinstance(k, str) and "loss" in k:
                try:
                    self.logger.info("step=%s %s=%.6f", step, k, float(v))
                except Exception:
                    self.logger.info("step=%s %s=%s", step, k, v)

@dataclass
class ModelArguments:

    model_name_or_path: Optional[str] = field(default="none", metadata={"help": ""})
    train_type: Optional[str] = field(
        default="none",
        metadata={
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": ""}
    )


def load_model_processor(modelargs: ModelArguments):


    model_cls = CustomLlavaForConditionalGeneration


    processor_cls = CustomLlavaProcessor
    processor = processor_cls.from_pretrained(modelargs.model_name_or_path, local_files_only=True)

    vocab_size = len(processor.tokenizer)

    if modelargs.train_type == "use_lora":
        model_path = modelargs.model_name_or_path + '/trained'
    else:
        model_path = modelargs.model_name_or_path
    model = model_cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,  
        low_cpu_mem_usage=True,  
        local_files_only=True   
    )
    print("modelargs.model_name_or_path:", modelargs.model_name_or_path)
    # for name, _ in model.named_modules():
    #     if "layers." in name:
    #         print(name)

    embed_size = model.get_input_embeddings().num_embeddings
    print(f"embed_size: {embed_size}, vocab_size: {vocab_size}")
    if embed_size != vocab_size:
        print(f">>> Resizing token embeddings: {embed_size} -> {vocab_size}")
        model.resize_token_embeddings(vocab_size)

    if hasattr(model, "config"):
        model.config.use_cache = False
    

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("tokenizer.pad_token set to tokenizer.eos_token")

    if modelargs.train_type == "use_lora":
        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model


        LORA_R = 16  
        LORA_ALPHA = 32  
        LORA_DROPOUT = 0.05  
        # TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = False
       
        layers = list(range(22, 28))       
        projs  = ["q_proj", "k_proj", "v_proj", "o_proj"]

        TARGET_MODULES = [
            f"model.language_model.layers.{layer}.self_attn.{proj}"
            for layer in layers
            for proj in projs
        ]

        # TARGET_MODULES = [
        #     "self_attn.q_proj",
        #     "self_attn.k_proj",
        #     "self_attn.v_proj",
        #     "self_attn.o_proj",
        # ]

        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",  
            task_type="CAUSAL_LM",  
        )


        model = get_peft_model(model, config)


        hit = []
        for n, m in model.named_modules():
            if any(t in n for t in TARGET_MODULES):
                hit.append((n, type(m).__name__))
        print(f"[LoRA] matched modules: {len(hit)}")
        for n, t in hit[:10]:
            print("  ", n, "->", t)
        if len(hit) > 0:
            print(f"[LoRA] total matched modules: {len(hit)}")
        else:
            raise ValueError("LoRA does not match any modules!")

    elif modelargs.train_type == "none":
        """Full parameter training"""
        logging.warning("Using full parameter training")

        pass
    elif modelargs.train_type == "freeze_vision":
        """Freeze all parameters of the vision tower"""
        logging.warning("Freezing vision_tower network layers, training the remaining network weights")
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    elif modelargs.train_type=="tune_mm_mlp_adapter":
        print(">>> Entered projector unfreeze branch <<<")
        model.requires_grad_(False)
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = True
            print(f"  Unfreeze: {p.shape}")
    # Print trainable parameter information
    print_trainable_parameters(model)

    return model, processor


def load_dataset_collator(processor, dataargs: DataArguments):
    """
    Load dataset and data collator
    :param processor: Model processor
    :param dataargs: Data argument configuration
    :return: Dataset and data collator objects
    """
    llava_dataset = LlavaDataset(
        dataargs.data_path  
    )

    logger.info(f"Loaded dataset from {dataargs.data_path}")

    # Initialize data collator
    data_collator = TrainLLavaModelCollator(
        processor=processor,
        IGNORE_INDEX=-100  # Index to ignore during loss calculation
    )

    return llava_dataset, data_collator


def train():
    """Main function for training the model"""
    # Initialize argument parser
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    # Parse command-line arguments
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()  

    # Prevent Trainer from dropping unused keys in the batch (e.g., pixel_values)
    training_args.remove_unused_columns = False
    # Load model and processor
    model, processor = load_model_processor(model_args)
    # Ensure compatibility with LoRA + gradient checkpointing
    if training_args.gradient_checkpointing:
        # Double insurance: enable gradient checkpointing
        model.gradient_checkpointing_enable()
        # Key: make input hidden states require gradients, otherwise treated as no_grad, causing loss to have no grad_fn
        try:
            model.enable_input_require_grads()
            print("Enabled enable_input_require_grads()")
        except AttributeError:
            # Llava top-level may not have this method, try the underlying language_model
            if hasattr(model, "language_model") and hasattr(model.language_model, "enable_input_require_grads"):
                model.language_model.enable_input_require_grads()
                print("Enabled language_model.enable_input_require_grads()")
    else:
        raise ValueError("Please enable gradient_checkpointing to save memory!")
    # Load dataset and data collator
    train_dataset, data_collator = load_dataset_collator(processor, data_args)
    # print(train_dataset[0])          # View the structure of the first sample
    trainer = Trainer(
        model=model,
        args=training_args,  # Training arguments
        train_dataset=train_dataset,
        eval_dataset=None,   # Temporarily not using validation set
        data_collator=data_collator,  # Pass in data collator
        callbacks=[LossLoggingCallback()],
    )
    # ========== Print trainable parameters ==========
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(f"{n:<60} {p.numel():>12,}")
    # ===================================
    # Start training
    logger.info("Training started...")
    trainer.train()
    logger.info("Training completed")

     # Save training state and model
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    # Configure logging format
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="pretrain.log",      # Added
        filemode="a"                  # Added
    )
    # Execute model training
    train()


