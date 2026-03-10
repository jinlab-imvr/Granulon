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

CLASS_NUM = 20

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
                t = t.unsqueeze(0)  # -> [1, patch, feature]
            else:
                print("Warning: Unexpected tensor shape in get_image_features:", t.shape)
                return t

            return t

        if isinstance(feats, list):
            feats = [fix_one(t) for t in feats]
        else:
            feats = fix_one(feats)
        # print("    fixed feats type:", type(feats))
        # print("    fixed feats:", feats if not isinstance(feats, list) else f"list of length {len(feats)}")
        # print("    fixed feats[0] shape:", feats[0].shape if isinstance(feats, list) and len(feats) > 0 else "N/A")
        return feats

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
