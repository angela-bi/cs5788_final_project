"""
Demo: stylized real-photo edits using Stable Diffusion img2img + Prompt-to-Prompt hooks.

This is additive to the original notebook workflow, which uses
`CompVis/ldm-text2im-large-256` (text2img). For real-photo editing, you need an SD checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
)

import ptp_utils


class EmptyControl:
    """No-op controller (keeps API consistent with P2P controllers)."""

    def step_callback(self, x_t: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x_t

    def between_steps(self) -> None:
        return

    def __call__(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        return attn


def _build_prompt(user_prompt: str, style: str) -> str:
    """
    Make the run less reliant on enumerating objects by injecting a style + composition hint.
    """
    user_prompt = (user_prompt or "").strip()
    style = (style or "").strip()
    if not style:
        return user_prompt

    # If the user already wrote a detailed prompt, just prepend the style.
    # If it's short (e.g. "make it watercolor"), still add a "same composition" hint.
    if len(user_prompt.split()) <= 6:
        return f"{style} watercolor painting of the same scene, same composition, preserve layout. {user_prompt}".strip()
    return f"{style} watercolor painting. {user_prompt}".strip()


def _default_negative_prompt() -> str:
    # Defaults tuned for "keep the photo's content, just change style".
    return (
        "low quality, blurry, jpeg artifacts, text, watermark, logo, "
        "extra objects, missing subject, different scene, rotated, tilted"
    )


def _resize_keep_aspect_multiple_of_8(img: Image.Image, max_side: int | None) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w0, h0 = img.size
    if max_side is not None and max(w0, h0) > max_side:
        scale = max_side / float(max(w0, h0))
        w0 = int(round(w0 * scale))
        h0 = int(round(h0 * scale))
    w = max((w0 // 8) * 8, 64)
    h = max((h0 // 8) * 8, 64)
    if (w, h) != img.size:
        img = img.resize((w, h), resample=Image.BICUBIC)
    return img


def _make_canny_control_image(img: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    # ControlNet canny expects a 3-channel image with edges in white.
    img_np = np.array(img)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, low_threshold, high_threshold)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_rgb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--controlnet",
        choices=["none", "canny"],
        default="none",
        help="Use ControlNet to better preserve composition.",
    )
    parser.add_argument(
        "--controlnet-id",
        default="lllyasviel/control_v11p_sd15_canny",
        help="ControlNet checkpoint id (only used when --controlnet != none).",
    )
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=1.0,
        help="How strongly ControlNet is enforced (0 disables).",
    )
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Prompt (can be short, e.g. 'make it watercolor')")
    parser.add_argument(
        "--style",
        default="",
        help="Optional style shortcut (e.g. 'soft', 'vibrant'). Always applies watercolor phrasing.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="__default__",
        help="Negative prompt (helps preserve key objects / avoid drift)",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="Cap max(H,W) before VAE encoding to reduce memory (set 0 to disable cap).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="out.png")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # MPS float16 can be unstable for diffusion; prefer float32 on Apple.
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    init_image = Image.open(args.image)
    init_image = _resize_keep_aspect_multiple_of_8(init_image, None if args.max_side == 0 else args.max_side)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    controller = EmptyControl()

    prompt = _build_prompt(args.prompt, args.style)
    negative_prompt = _default_negative_prompt() if args.negative_prompt == "__default__" else args.negative_prompt

    if args.controlnet == "canny":
        controlnet = ControlNetModel.from_pretrained(args.controlnet_id, torch_dtype=dtype).to(device)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            args.model_id, controlnet=controlnet, torch_dtype=dtype
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        # Patch attention (so Prompt-to-Prompt controllers still work).
        ptp_utils.register_attention_control(pipe, controller)

        control_image = _make_canny_control_image(init_image, args.canny_low, args.canny_high)
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            control_image=control_image,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            controlnet_conditioning_scale=args.controlnet_scale,
        )
        images = [np.array(out.images[0])]
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        images, _latents = ptp_utils.image2image_ldm_stable(
            pipe,
            prompt=[prompt],
            controller=controller,
            init_image=init_image,
            strength=args.strength,
            negative_prompt=[negative_prompt],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            size=None,
            fit="pad",
            max_side=None if args.max_side == 0 else args.max_side,
        )

    out_path = Path(args.out)
    Image.fromarray(images[0]).save(out_path)
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()

