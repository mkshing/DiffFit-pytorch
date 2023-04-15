# DiffFit-pytorch
<a href="https://colab.research.google.com/github/mkshing/difffit-pytorch/blob/main/scripts/difffit_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


An implementation of [DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2304.06648) by using d🧨ffusers. 

My summary tweet is found [here](https://twitter.com/mk1stats/status/1647246562765012993).

⚠️ This is working in progress. You might need to adjust hyper-params, especially the learning rate. 

<!-- ![result]
left: LoRA, right: SVDiff
Compared with LoRA, the number of trainable parameters is 0.22 M less parameters and the file size is only 2.4 MB (LoRA: 3.1 MB)!! -->

## What is DiffFit?
DiffFit is an extension of an existing PEFT called [BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models](https://arxiv.org/abs/2106.10199) where only the bias-terms of the model are trainable. In addition to bias-terms, learnable scaling factors to each transformer block and LN are trained for better adaption. 

<img src="difffit-overview.png" alt= "overview" width="400">

In the paper, they applied DiffFit to [DiT](https://www.wpeebles.com/DiT) instead of UNet-based DMs but mentioned DiffFit can be generalized to other DMs e.g. text-to-image models like Stable Diffusion. So, in this repo, I extended it for Stable Diffusion in domain-tuning setting. 
- Add scale factors in transformer-blocks and attention blocks including cross-attention
- Adapt DiffFit to text encoder
- Introduce the strategy (identifier token, prior preservation) of Dreambooth

Also, you can try [BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models](https://arxiv.org/abs/2106.10199) here. 


## Installation 
```bash
$ git clone https://github.com/mkshing/DiffFit-pytorch.git
$ pip install -r requirements.txt
```


## Training
<!-- For learning rate, they mentioned that:
> We observe that using a learning rate 10× greater than pre-training yields the best result. -->
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_difffit.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-3 \
  --add_vlb_loss \
  --vlb_lambda=0.001 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=500 \
  # you can add prior preservation loss
  # --class_prompt="a photo of dog" \
  # --with_prior_preservation --prior_loss_weight=1.0 \
  # optionally add extra params in text encoder
  # --train_text_encoder \
```

Add `--bitfit` for BitFit. 

## Inference
```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from difffit_pytorch import load_unet_for_difffit, load_text_encoder_for_difffit, load_config_for_difffit

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
efficient_weights_ckpt = "ckpt-dir-path"

training_args = load_config_for_difffit(args.efficient_weights_ckpt)
unet = load_unet_for_difffit(args.pretrained_model_name_or_path, efficient_weights_ckpt=args.efficient_weights_ckpt, is_bitfit=training_args["bitfit"], subfolder="unet")
text_encoder = load_text_encoder_for_difffit(args.pretrained_model_name_or_path, efficient_weights_ckpt=args.efficient_weights_ckpt, is_bitfit=training_args["bitfit"], subfolder="text_encoder")

# load pipe
pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    unet=unet,
    text_encoder=text_encoder,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
```
