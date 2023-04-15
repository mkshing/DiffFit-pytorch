import os
from typing import Dict
import inspect
import json
import torch
from torch import nn
import accelerate
from accelerate.utils import set_module_tensor_to_device
from transformers import CLIPTextModel, CLIPTextConfig
from diffusers import UNet2DConditionModel
from safetensors.torch import safe_open
import huggingface_hub
from difffit_pytorch.diffusers_models.unet_2d_condition import UNet2DConditionModel as UNet2DConditionModelForDiffFit
from difffit_pytorch.diffusers_models.attention_processor import DiffFitAttnProcessor, DiffFitXFormersAttnProcessor
from difffit_pytorch.transformers_models_clip.modeling_clip import CLIPTextModel as CLIPTextModelForDiffFit


def mark_only_biases_as_trainable(model: nn.Module, is_bitfit=False):
    if is_bitfit:
        trainable_names = ['bias']
    else:
        trainable_names = ["bias","norm","gamma","y_embed"]

    for par_name, par_tensor in model.named_parameters():
        par_tensor.requires_grad = any([kw in par_name for kw in trainable_names])
    return model


def get_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        par_name: par_tensor
        for par_name, par_tensor in model.named_parameters()
        if par_tensor.requires_grad
    }


def load_config_for_difffit(model_path, **kwargs):
    if os.path.exists(model_path):
        if "config.json" not in model_path:
            model_path = os.path.join(model_path, "config.json")
    else:
        model_path = huggingface_hub.hf_hub_download(model_path, filename="config.json", **kwargs)
    with open(model_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config



def load_unet_for_difffit(
        pretrained_model_name_or_path,
        efficient_weights_ckpt=None, 
        hf_hub_kwargs=None, 
        is_bitfit=False, 
        **kwargs
    ):
    """
    https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/models/modeling_utils.py#L541
    """
    # load pre-trained weights
    param_device = "cpu"
    torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None
    if not is_bitfit:
        config = UNet2DConditionModel.load_config(pretrained_model_name_or_path, **kwargs)
        original_model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        state_dict = original_model.state_dict()
        with accelerate.init_empty_weights():
            model = UNet2DConditionModelForDiffFit.from_config(config)
            # Set correct lora layers
            difffit_attn_procs = {}
            for name in model.attn_processors.keys():
                if name.startswith("mid_block"):
                    hidden_size = model.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(model.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = model.config.block_out_channels[block_id]

                difffit_attn_procs[name] = DiffFitAttnProcessor(hidden_size=hidden_size)

            model.set_attn_processor(difffit_attn_procs)
            
        scale_factor_weights = {n: torch.ones(p.shape) for n, p in model.named_parameters() if "gamma_" in n}
        state_dict.update(scale_factor_weights)
        # move the params from meta device to cpu
        missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_keys) > 0:
            raise ValueError(
                f"Cannot load {model.__class__.__name__} from {pretrained_model_name_or_path} because the following keys are"
                f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize"
                " those weights or else make sure your checkpoint file is correct."
            )
        for param_name, param in state_dict.items():
            accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
            if accepts_dtype:
                set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=torch_dtype)
            else:
                set_module_tensor_to_device(model, param_name, param_device, value=param)
    else:
        original_model = None
        model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, **kwargs)

    if efficient_weights_ckpt:
        if os.path.isdir(efficient_weights_ckpt):
            efficient_weights_ckpt = os.path.join(efficient_weights_ckpt, "efficient_weights.safetensors")
        elif not os.path.exists(efficient_weights_ckpt):
            # download from hub
            hf_hub_kwargs = {} if hf_hub_kwargs is None else hf_hub_kwargs
            efficient_weights_ckpt = huggingface_hub.hf_hub_download(efficient_weights_ckpt, filename="efficient_weights.safetensors", **hf_hub_kwargs)
        assert os.path.exists(efficient_weights_ckpt)

        with safe_open(efficient_weights_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                # spectral_shifts_weights[key] = f.get_tensor(key)
                accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                if accepts_dtype:
                    set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key), dtype=torch_dtype)
                else:
                    set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key))
        print(f"Resumed from {efficient_weights_ckpt}")
    if "torch_dtype"in kwargs:
        model = model.to(kwargs["torch_dtype"])
    model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    del original_model
    torch.cuda.empty_cache()
    return model



def load_text_encoder_for_difffit(
        pretrained_model_name_or_path,
        efficient_weights_ckpt=None,
        hf_hub_kwargs=None,
        is_bitfit=False,
        **kwargs
    ):
    """
    https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/models/modeling_utils.py#L541
    """
    # load pre-trained weights
    param_device = "cpu"
    torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None
    if not is_bitfit:
        config = CLIPTextConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        original_model = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        state_dict = original_model.state_dict()
        with accelerate.init_empty_weights():
            model = CLIPTextModelForDiffFit(config)
        scale_factor_weights = {n: torch.ones(p.shape) for n, p in model.named_parameters() if "gamma_" in n}
        state_dict.update(scale_factor_weights)
        # move the params from meta device to cpu
        missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_keys) > 0:
            raise ValueError(
                f"Cannot load {model.__class__.__name__} from {pretrained_model_name_or_path} because the following keys are"
                f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize"
                " those weights or else make sure your checkpoint file is correct."
            )

        for param_name, param in state_dict.items():
            accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
            if accepts_dtype:
                set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=torch_dtype)
            else:
                set_module_tensor_to_device(model, param_name, param_device, value=param)
    else:
        original_model = None
        model = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, **kwargs)

    if efficient_weights_ckpt:
        if os.path.isdir(efficient_weights_ckpt):
            efficient_weights_ckpt = os.path.join(efficient_weights_ckpt, "efficient_weights_te.safetensors")
        elif not os.path.exists(efficient_weights_ckpt):
            # download from hub
            hf_hub_kwargs = {} if hf_hub_kwargs is None else hf_hub_kwargs
            try:
                efficient_weights_ckpt = huggingface_hub.hf_hub_download(efficient_weights_ckpt, filename="efficient_weights_te.safetensors", **hf_hub_kwargs)
            except huggingface_hub.utils.EntryNotFoundError:
                efficient_weights_ckpt = None
        # load state dict only if `spectral_shifts_te.safetensors` exists
        if os.path.exists(efficient_weights_ckpt):
            with safe_open(efficient_weights_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # spectral_shifts_weights[key] = f.get_tensor(key)
                    accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                    if accepts_dtype:
                        set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key), dtype=torch_dtype)
                    else:
                        set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key))
            print(f"Resumed from {efficient_weights_ckpt}")
        
    if "torch_dtype"in kwargs:
        model = model.to(kwargs["torch_dtype"])
    # model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    del original_model
    torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    unet = load_unet_for_difffit("CompVis/stable-diffusion-v1-4", subfolder="unet")
    for n, p in unet.named_parameters():
        if "gamma" in n:
            print(n)
