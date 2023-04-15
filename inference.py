import argparse
from tqdm import tqdm
import random
from PIL import Image
import torch
from diffusers import (
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionPipeline
)
from diffusers.utils import is_xformers_available
from difffit_pytorch.utils import load_unet_for_difffit, load_text_encoder_for_difffit, load_config_for_difffit


SCHEDULER_MAPPING = {
    "ddim": DDIMScheduler,
    "plms": PNDMScheduler,
    "lms": LMSDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm_solver++": DPMSolverMultistepScheduler,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="pretrained model name or path")
    parser.add_argument("--efficient_weights_ckpt", type=str, help="path to efficient_weights.safetensors")
    # diffusers config
    parser.add_argument("--prompt", type=str, nargs="?", default="a photo of *s", help="the prompt to render")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="number of sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="unconditional guidance scale")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="number of images per prompt")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space",)
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space",)
    parser.add_argument("--seed", type=str, default="random_seed", help="the seed (for reproducible sampling)")
    parser.add_argument("--scheduler_type", type=str, choices=["ddim", "plms", "lms", "euler", "euler_ancestral", "dpm_solver++"], default="ddim", help="diffusion scheduler type")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--spectral_shifts_scale", type=float, default=1.0, help="scaling spectral shifts")
    parser.add_argument("--fp16", action="store_true", help="fp16 inference")
    args = parser.parse_args()
    return args


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    training_args = load_config_for_difffit(args.efficient_weights_ckpt)
    unet = load_unet_for_difffit(args.pretrained_model_name_or_path, efficient_weights_ckpt=args.efficient_weights_ckpt, is_bitfit=training_args["bitfit"], subfolder="unet")
    text_encoder = load_text_encoder_for_difffit(args.pretrained_model_name_or_path, efficient_weights_ckpt=args.efficient_weights_ckpt, is_bitfit=training_args["bitfit"], subfolder="text_encoder")

    # load pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        requires_safety_checker=False,
        safety_checker=None,
        feature_extractor=None,
        scheduler=SCHEDULER_MAPPING[args.scheduler_type].from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
        torch_dtype=torch.float16 if args.fp16 else None,
    )
    if args.enable_xformers_memory_efficient_attention:
        assert is_xformers_available()
        pipe.enable_xformers_memory_efficient_attention()
        print("Using xformers!")
    try:
        import tomesd
        tomesd.apply_patch(pipe, ratio=0.5)
        print("Using tomesd!")
    except:
        pass
    pipe = pipe.to(device)
    print("loaded pipeline")
    # run!    
    if args.seed == "random_seed":
        random.seed()
        seed = random.randint(0, 2**32)
    else:
        seed = int(args.seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"seed: {seed}")
    prompts = args.prompt.split("::")
    all_images = []
    for prompt in tqdm(prompts):
        with torch.autocast(device), torch.inference_mode():
            images = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                num_images_per_prompt=args.num_images_per_prompt,
                height=args.height,
                width=args.width,
            ).images
        all_images.extend(images)
    grid_image = image_grid(all_images, len(prompts), args.num_images_per_prompt)
    grid_image.save("grid.png")
    print("DONE! See `grid.png` for the results!")


if __name__ == '__main__':
    main()

