import torch.nn as nn

def get_nb_trainable_parameters(model:nn.Module) -> tuple[int, int]:
    """
    return the number of trainable parameters in the model

    parameters:
        model (nn.Module): the model to count parameters for

    returns:
        tuple[int, int]: a tuple containing the number of trainable parameters and the total number of parameters
    """
    # Initialize the number of trainable parameters
    trainable_params = 0
    # Initialize the total number of parameters
    all_param = 0
    # Iterate over all named parameters in the model
    for _, param in model.named_parameters():
        # Get the number of elements in the current parameter
        num_params = param.numel()
        # If using DeepSpeed Zero 3 and the weights are initialized as empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            # Use the number of elements counted by DeepSpeed
            num_params = param.ds_numel

        # Due to the design of 4-bit linear layers in the bitsandbytes library
        # the number of parameters needs to be multiplied by 2 to get the correct count
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                # Get the size of each element in bytes
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                # If there is no quantization storage attribute, default byte size is 1
                num_bytes = 1
            else:
                # Get the size of the quantization storage in bytes
                num_bytes = param.quant_storage.itemsize
            # Adjust the number of parameters
            num_params = num_params * 2 * num_bytes

        # Accumulate the total number of parameters
        all_param += num_params
        # If the current parameter requires gradient updates
        if param.requires_grad:
            # Accumulate the number of trainable parameters
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.

    Attention: print_trainable_parameters() uses get_nb_trainable_parameters(), which differs from
    num_parameters(only_trainable=True) in huggingface/transformers. get_nb_trainable_parameters() returns
    (trainable parameters, all parameters) for the Peft model, including the modified backbone transformer model.
    For techniques like LoRA, the backbone transformer model is modified in place by the LoRA modules.
    However, for prompt tuning, the backbone transformer model is not modified. num_parameters(only_trainable=True)
    returns the number of trainable parameters of the backbone transformer model, which may differ.
    Parameters:
        model (nn.Module): the model to print trainable parameter information for
    """
    # Get the number of trainable parameters and all parameters
    trainable_params, all_param = get_nb_trainable_parameters(model)

    # Print the number of trainable parameters, all parameters, and the percentage of trainable parameters
    print(
        f"Trainable parameters: {trainable_params:,d} || All parameters: {all_param:,d} || Trainable percentage: {100 * trainable_params / all_param:.4f}"
    )


if __name__ == "__main__":
    from transformers import LlavaForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    import torch

    # Load LLaVA model
    model_name_or_path = ""
    model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        torch_dtype=torch.bfloat16,  # Use BF16 precision
        low_cpu_mem_usage=True,  # Optimize CPU memory usage
        local_files_only=True   # Use local files only
    )

    # LoRA parameter configuration
    LORA_R = 32  # Rank parameter, controls the dimension of low-rank approximation
    # LORA_ALPHA = 16  # Scaling factor, used to adjust the magnitude of weight updates in LoRA modules
    LORA_DROPOUT = 0.05  # Dropout rate
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Modules to apply LoRA to

    # Initialize LoRA configuration
    config = LoraConfig(
        r=LORA_R,
        # lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",  # LoRA only affects the weight matrices and does not affect the bias terms
        task_type="CAUSAL_LM",  # Indicates the task type is causal language modeling, i.e., predicting the next word based on previous text
        modules_to_save=["multi_modal_projector"],  # Indicates that when saving the model, in addition to LoRA-related parameters, the parameters of the module named multi_modal_projector also need to be saved
    )

    # Apply LoRA to the model
    model = get_peft_model(model, config)

    # Print trainable parameter information
    print_trainable_parameters(model)
