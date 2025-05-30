import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from ..utils import PeftConfig, PeftType, transpose
from loguru import logger as log

@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """
    hidden_size: int = field(default=1536)
    num_attention_heads: int = field(default=12)
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        }
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_nums: int = field(default=None, metadata={"help": "Numbers of Lora"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model): # LoraConfig, CasualLM
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit):
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "lora_nums": self.peft_config.lora_nums,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
            "hidden_size": self.peft_config.hidden_size,
            "num_attention_heads": self.peft_config.num_attention_heads
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found: # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                # print(f"self.peft_config.enable_lora:{self.peft_config.enable_lora}")
                if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)

                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False 
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

class CrossAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.hidden_size = kwargs["hidden_size"]
        self.num_heads = kwargs["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5  # Pre-compute scaling factor
        self.output = torch.tensor(0)
        self.lora_q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.lora_k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.lora_v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.lora_o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Add attention dropout
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, mask=None):
        if not mask:
            return self.output
        # Get batch size and sequence length
        batch_size, seq_len, _ = query.shape
        
        # Linear projections and reshape
        q = self.lora_q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.lora_k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.lora_v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.silu(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.lora_o_proj(output)
        output = self.resid_dropout(output)
        return output
        
class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        
        self.AttentionModule = CrossAttention(**kwargs)
        self.lora_num = lora_nums
        self.fan_in_fan_out = fan_in_fan_out
        self.relative_score = 1

        # Actual trainable parameters
        if r > 0:
            for i in range(self.lora_num):
                setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        
        if hasattr(self, "lora_A0"):
            for i in range(self.lora_num):
                nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B{i}").weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").train(mode)
            getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").eval()
            getattr(self, f"lora_B{i}").eval()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)

    def compute_cross_att(main_lora, other_loras):
        pass
    
    def getRelavance(main_lora: torch.tensor, other_loras: list[torch.tensor]):
        # Compute cosine similarity between main_lora and each other_lora
        main_lora_flat = main_lora.view(-1)
        similarities = []
        
        for other_lora in other_loras:
            other_lora_flat = other_lora.view(-1)
            
            # Compute cosine similarity
            dot_product = torch.dot(main_lora_flat, other_lora_flat)
            main_norm = torch.norm(main_lora_flat)
            other_norm = torch.norm(other_lora_flat)
            
            # Add small epsilon to avoid division by zero
            similarity = dot_product / (main_norm * other_norm + 1e-8)
            similarities.append(similarity)
            
        return torch.tensor(similarities, device=main_lora.device)

    
    def forward(self, x: torch.Tensor, task_idx=None):
        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError(":(")
        elif self.r > 0 and not self.merged:
            raw_ffn = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            result = raw_ffn

            main_lora = getattr(self, f"lora_B{task_idx[0].item()}")(getattr(self, f"lora_A{task_idx[0].item()}")(self.lora_dropout(x))) * self.scaling

            other_loras = [
                getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling
                for i in range(self.lora_num) if i != task_idx[0].item()
            ]
            
            for idx, lora in enumerate(other_loras):
                result += self.AttentionModule(main_lora, lora, lora)

            result += main_lora
            
        return result