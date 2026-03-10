from typing import Any
import torch
import torch.nn as nn

import numpy as np

try:
    from transformers.models.llava.modeling_llava import (
        LlavaModel as HF_LlavaModel,
        LlavaForConditionalGeneration as HF_LlavaForConditionalGeneration,
        LlavaPreTrainedModel,
    )
except Exception:
    from transformers import (
        LlavaModel as HF_LlavaModel,
        LlavaForConditionalGeneration as HF_LlavaForConditionalGeneration,
        LlavaPreTrainedModel,
    )

# Processor imports across versions
try:
    from transformers.models.llava.processing_llava import (
        LlavaProcessor as HF_LlavaProcessor,
        LlavaProcessorKwargs,
    )
except Exception:
    from transformers import LlavaProcessor as HF_LlavaProcessor  # type: ignore
    try:
        from transformers.models.llava.processing_llava import LlavaProcessorKwargs  # type: ignore
    except Exception:
        LlavaProcessorKwargs = None  # type: ignore

try:
    from transformers.feature_extraction_utils import BatchFeature
except Exception:
    BatchFeature = dict  # very old versions fallback

from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from typing import Optional, Union

CLASS_NUM = 5  

class LlavaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "return_mm_token_type_ids": False},
        "images_kwargs": {},
    }

class CustomLlavaModel(HF_LlavaModel):

    def get_image_features(self, *args: Any, **kwargs: Any):
        feats = super().get_image_features(*args, **kwargs)
        # print(">>> CustomLlavaModel.get_image_features called, original feats type:", type(feats))
        # print("    original feats:", feats if not isinstance(feats, list) else f"list of length {len(feats)}")
        # print("    original feats[0] shape:", feats[0].shape if isinstance(feats, list) and len(feats) > 0 else "N/A")
        def fix_one(t: torch.Tensor) -> torch.Tensor:
            if not isinstance(t, torch.Tensor):
                return t
            if t.dim() == 2:
                if t.size(0) >= 5:
                    t = t[5:, :]
                t = t.unsqueeze(0)  # -> [1, patch, feature]
            else:
                print("Warning: Unexpected tensor shape in get_image_features:", t.shape)
                return t

            return t
        
        if isinstance(feats, list):
            feats = [fix_one(t) for t in feats]
            features = []
            for i in range(len(feats)):

                patches = torch.nn.functional.normalize(feats[i], dim=-1)

                sim_matrix = patches @ patches.transpose(-1, -2)  


                X = sim_matrix[0]  
                n_clusters = CLASS_NUM
                device = X.device
                torch.manual_seed(42)  

                def kmeans_pytorch(X: torch.Tensor, num_clusters: int, max_iter: int = 100, tol: float = 1e-6):
                    N, D = X.shape
                    device = X.device
                    torch.manual_seed(42)

                    with torch.no_grad():                       

                        idx = torch.randperm(N, device=device)[:num_clusters]
                        centers = X[idx]                        # [K, D]

                        for _ in range(max_iter):              

                            dists = torch.cdist(X, centers, p=2)
                            labels = dists.argmin(dim=1)

                            one_hot = torch.zeros(N, num_clusters, device=device, dtype=X.dtype)
                            one_hot.scatter_(1, labels.view(-1, 1), 1)

                            new_centers = one_hot.t() @ X
                            counts = one_hot.sum(0).clamp_min(1)
                            new_centers /= counts.view(-1, 1)

                            if torch.allclose(centers, new_centers, atol=tol):
                                break
                            centers = new_centers

                    return centers, labels

                device = torch.cuda.current_device()        
                X = sim_matrix[0].to(device, non_blocking=True)
                cluster_centers, labels_ = kmeans_pytorch(X, n_clusters, max_iter=100)
                
                cluster_feat = torch.zeros(n_clusters, feats[i].shape[2], device=device)
                for k in range(n_clusters):
                    mask = (labels_ == k)
                    if mask.sum() == 0:
                        continue
                    cluster_feat[k] = feats[i][0, mask].mean(dim=0)  
                cluster_feat = cluster_feat.unsqueeze(0)  # [1, n_clusters, D]
                new_feats = torch.cat((feats[i], cluster_feat), dim=1)  # [1, 196+n_clusters, D]
                features.append(new_feats)
            return features


        else:
            raise TypeError("feats is not a list, unexpected.")
        # print("    fixed feats type:", type(feats))
        # print("    fixed feats:", feats if not isinstance(feats, list) else f"list of length {len(feats)}")
        # print("    fixed feats[0] shape:", feats[0].shape if isinstance(feats, list) and len(feats) > 0 else "N/A")


class CustomLlavaForConditionalGeneration(HF_LlavaForConditionalGeneration):

    _checkpoint_conversion_mapping = getattr(
        HF_LlavaForConditionalGeneration, "_checkpoint_conversion_mapping", {}
    )
    _tied_weights_keys = getattr(
        HF_LlavaForConditionalGeneration, "_tied_weights_keys", []
    )

    def __init__(self, config):
        LlavaPreTrainedModel.__init__(self, config)
        self.model = CustomLlavaModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

class CustomLlavaProcessor(HF_LlavaProcessor):

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[LlavaProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            LlavaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            # Replace the image token with the expanded image token sequence
            pixel_values = image_inputs["pixel_values"]
            height, width = get_image_size(to_numpy_array(pixel_values[0]))
            num_image_tokens = (height // self.patch_size) * (
                width // self.patch_size
            ) + self.num_additional_image_tokens
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            num_image_tokens += CLASS_NUM

            prompt_strings = []
            for sample in text:
                sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
                prompt_strings.append(sample)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)
