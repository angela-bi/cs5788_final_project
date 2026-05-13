"""
DDIM inversion, null-text optimization, and prompt-to-prompt editing on real photos
(Stable Diffusion v1.5). Call init_project(model) once after loading the pipeline.
"""

from __future__ import annotations

import gc
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
from torch.optim import Adam
from tqdm.auto import tqdm

import ptp_utils
from p2p_controllers import (
    AttentionRefine,
    AttentionReplace,
    AttentionReweight,
    AttentionStore,
    LocalBlend,
    P2PRuntime,
    get_runtime,
    make_controller,
    set_runtime,
)

# Re-export controller helpers for notebooks: `import real_image_edit as re`
__all__ = [
    "init_project",
    "load_512",
    "image_to_latent",
    "get_text_embeddings",
    "ddim_inversion",
    "edit_real_image",
    "NullInversion",
    "edit_real_image_null",
    "visualize_noising_process",
    "visualize_null_text_optimization",
    "aggregate_attention",
    "show_cross_attention",
    "show_self_attention_comp",
    "LocalBlend",
    "AttentionRefine",
    "AttentionReplace",
    "AttentionReweight",
    "make_controller",
    "get_runtime",
]


def init_project(
    model,
    num_ddim_steps: int = 50,
    guidance_scale: float = 7.5,
    max_num_words: int = 77,
) -> None:
    """Register the active SD pipeline and P2P hyperparameters (tokenizer, device, steps)."""
    set_runtime(P2PRuntime(model, num_ddim_steps, guidance_scale, max_num_words))
    model.scheduler.set_timesteps(num_ddim_steps)


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if isinstance(image_path, str):
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top : h - bottom, left : w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset : offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def image_to_latent(vae, image: Union[np.ndarray, Image.Image], device="cuda") -> torch.Tensor:
    img_tensor = torch.tensor(np.array(image), dtype=torch.float32).to(device)
    img_tensor = img_tensor / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(dtype=next(vae.parameters()).dtype)
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.mean
        latent = latent * 0.18215
    del img_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return latent


def get_text_embeddings(model, prompt: str) -> torch.Tensor:
    text_input = model.tokenizer(
        [prompt],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return embeddings


def _ddim_inversion_step(scheduler, noise_pred: torch.Tensor, t: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    t_int = int(t)
    alpha_prod_t = scheduler.alphas_cumprod[t_int]
    next_t = min(
        t_int + scheduler.config.num_train_timesteps // scheduler.num_inference_steps,
        scheduler.config.num_train_timesteps - 1,
    )
    alpha_prod_next = scheduler.alphas_cumprod[next_t]
    x0_pred = (latent - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t**0.5
    next_latent = alpha_prod_next**0.5 * x0_pred + (1 - alpha_prod_next) ** 0.5 * noise_pred
    return next_latent


def ddim_inversion(
    model,
    image: Union[np.ndarray, Image.Image],
    source_prompt: str,
    num_inference_steps: Optional[int] = None,
    guidance_scale: float = 1.0,
) -> List[torch.Tensor]:
    if num_inference_steps is None:
        num_inference_steps = get_runtime().num_ddim_steps
    latent = image_to_latent(model.vae, image, device=model.device)
    text_embeddings = get_text_embeddings(model, source_prompt)
    uncond_input = model.tokenizer(
        [""],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    model.scheduler.set_timesteps(num_inference_steps)
    timesteps = model.scheduler.timesteps

    all_latents = [latent.clone()]
    for _i, t in enumerate(
        tqdm(reversed(timesteps), desc="DDIM Inversion", total=num_inference_steps)
    ):
        with torch.no_grad():
            if guidance_scale != 1.0:
                latent_input = torch.cat([latent] * 2)
                context = torch.cat([uncond_embeddings, text_embeddings])
                noise_pred = model.unet(latent_input, t, encoder_hidden_states=context)["sample"]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = model.unet(latent, t, encoder_hidden_states=text_embeddings)["sample"]
        latent = _ddim_inversion_step(model.scheduler, noise_pred, t, latent)
        all_latents.append(latent.clone())
    return all_latents


@torch.no_grad()
def edit_real_image(
    model,
    controller,
    image: Union[str, np.ndarray, Image.Image],
    source_prompt: str,
    target_prompt: str,
    num_inference_steps: Optional[int] = None,
    guidance_scale: float = 7.5,
    inversion_guidance_scale: float = 1.0,
    low_resource: bool = False,
) -> dict:
    if num_inference_steps is None:
        num_inference_steps = get_runtime().num_ddim_steps
    if isinstance(image, str):
        image = load_512(image)
        image = np.rot90(image, k=3, axes=(0, 1))
    else:
        image = np.array(image)
    original_image = image.copy()

    print("inverting image with DDIM")
    all_latents = ddim_inversion(
        model,
        image,
        source_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=inversion_guidance_scale,
    )
    xT = all_latents[-1]

    print("editing with P2P")
    edited_images, _ = ptp_utils.text2image_ldm_stable(
        model=model,
        prompt=[source_prompt, target_prompt],
        controller=controller,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        latent=xT,
        low_resource=low_resource,
    )
    return {
        "original": original_image,
        "reconstruction": edited_images[0],
        "edited": edited_images[1],
        "latents": all_latents,
    }


class NullInversion:
    def prev_step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor):
        rt = get_runtime()
        prev_timestep = timestep - rt.model.scheduler.config.num_train_timesteps // rt.model.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor):
        timestep, next_timestep = (
            min(
                timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps,
                999,
            ),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        return self.model.unet(latents, t, encoder_hidden_states=context)["sample"]

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        rt = get_runtime()
        guidance_scale = 1 if is_forward else rt.guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def image2latent(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            return image
        rt = get_runtime()
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(rt.device)
        latents = self.model.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        _uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        rt = get_runtime()
        for i in range(rt.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def null_optimization(self, latents, num_inner_steps, epsilon):
        rt = get_runtime()
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * rt.num_ddim_steps)
        for i in range(rt.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1.0 - i / 100.0))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            j = 0
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + rt.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for _ in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list

    def __init__(self, model):
        self.model = model
        self.tokenizer = self.model.tokenizer
        rt = get_runtime()
        self.model.scheduler.set_timesteps(rt.num_ddim_steps)
        self.prompt = None
        self.context = None


def edit_real_image_null(
    model,
    controller,
    image: Union[str, np.ndarray, Image.Image],
    source_prompt: str,
    target_prompt: str,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_inner_steps: int = 10,
    early_stop_epsilon: float = 1e-5,
    offsets: Tuple[int, int, int, int] = (0, 0, 0, 0),
    low_resource: bool = False,
) -> dict:
    _ = low_resource  # API compatibility with the old notebook signature
    rt = get_runtime()
    if num_inference_steps is None:
        num_inference_steps = rt.num_ddim_steps
    if guidance_scale is None:
        guidance_scale = rt.guidance_scale

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if isinstance(image, str):
        image = load_512(image, *offsets)
        image = np.rot90(image, k=3, axes=(0, 1))

    print("ddim inversion")
    ptp_utils.register_attention_control(model, None)
    all_latents = ddim_inversion(
        model=model,
        image=image,
        source_prompt=source_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
    )
    xT = all_latents[-1]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        image_rec = ptp_utils.latent2image(model.vae, all_latents[0].detach())

    print("null inversion")
    null_inversion = NullInversion(model)
    null_inversion.init_prompt(source_prompt)
    uncond_embeddings = null_inversion.null_optimization(
        latents=all_latents,
        num_inner_steps=num_inner_steps,
        epsilon=early_stop_epsilon,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("editing with p2p")
    controller.reset()
    ptp_utils.register_attention_control(model, controller)

    src_emb = get_text_embeddings(model, source_prompt)
    tgt_emb = get_text_embeddings(model, target_prompt)
    text_embeddings = torch.cat([src_emb, tgt_emb])
    latents = xT.expand(2, -1, -1, -1).clone()
    model.scheduler.set_timesteps(num_inference_steps)

    with torch.no_grad():
        for i, t in enumerate(tqdm(model.scheduler.timesteps, desc="Denoising")):
            uncond_emb = uncond_embeddings[i].expand(2, -1, -1)
            context = torch.cat([uncond_emb, text_embeddings])
            latents_input = torch.cat([latents] * 2)
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
            latents = controller.step_callback(latents)

    with torch.no_grad():
        edited_images = ptp_utils.latent2image(model.vae, latents)

    return {
        "original": image,
        "reconstruction": image_rec,
        "edited": edited_images[1],
        "latents": all_latents,
    }


def visualize_noising_process(
    model,
    image: Union[str, np.ndarray, Image.Image],
    source_prompt: str,
    num_inference_steps: Optional[int] = None,
    num_vis_steps: int = 10,
) -> list:
    if num_inference_steps is None:
        num_inference_steps = get_runtime().num_ddim_steps
    if isinstance(image, str):
        image = load_512(image)
        image = np.rot90(image, k=3, axes=(0, 1))

    ptp_utils.register_attention_control(model, None)
    all_latents = ddim_inversion(
        model=model,
        image=image,
        source_prompt=source_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
    )
    total = len(all_latents)
    indices = np.linspace(0, total - 1, num_vis_steps, dtype=int)
    snapshots = []
    for idx in indices:
        with torch.no_grad():
            img = ptp_utils.latent2image(model.vae, all_latents[idx].detach())
        snapshots.append(img[0] if img.ndim == 4 else img)

    labeled = []
    for idx, snap in zip(indices, snapshots):
        noise_frac = idx / (total - 1)
        label = f"t={noise_frac:.1%}"
        labeled.append(ptp_utils.text_under_image(snap, label))

    ptp_utils.view_images(labeled, num_rows=1, offset_ratio=0.02)
    return labeled


def visualize_null_text_optimization(
    model,
    image: Union[str, np.ndarray, Image.Image],
    source_prompt: str,
    num_inference_steps: Optional[int] = None,
    num_inner_steps: int = 10,
    early_stop_epsilon: float = 1e-5,
    num_vis_steps: int = 10,
    denoise_guidance_scale: float = 7.5,
) -> None:
    if num_inference_steps is None:
        num_inference_steps = get_runtime().num_ddim_steps
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if isinstance(image, str):
        image = load_512(image)
        image = np.rot90(image, k=3, axes=(0, 1))

    print("Step 1/2 — DDIM inversion...")
    ptp_utils.register_attention_control(model, None)
    all_latents = ddim_inversion(
        model=model,
        image=image,
        source_prompt=source_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
    )

    print("Step 2/2 — Null-text optimization...")
    null_inversion = NullInversion(model)
    null_inversion.init_prompt(source_prompt)
    uncond_embeddings = null_inversion.null_optimization(
        latents=all_latents,
        num_inner_steps=num_inner_steps,
        epsilon=early_stop_epsilon,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.scheduler.set_timesteps(num_inference_steps)
    timesteps = model.scheduler.timesteps
    text_embeddings = get_text_embeddings(model, source_prompt)

    total = len(all_latents)
    indices = np.linspace(1, total - 1, num_vis_steps, dtype=int)
    rows = []
    for latent_idx in indices:
        denoise_step_idx = num_inference_steps - latent_idx
        denoise_step_idx = max(0, min(denoise_step_idx, num_inference_steps - 1))
        t = timesteps[denoise_step_idx]
        noise_frac = latent_idx / (total - 1)
        label_t = f"t={noise_frac:.0%}"
        noisy_latent = all_latents[latent_idx].detach()

        with torch.no_grad():
            before_img = ptp_utils.latent2image(model.vae, noisy_latent)[0]

        with torch.no_grad():
            uncond_emb = uncond_embeddings[denoise_step_idx]
            context = torch.cat([uncond_emb, text_embeddings])
            latent_input = torch.cat([noisy_latent] * 2)
            noise_pred = model.unet(latent_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_guided = noise_pred_uncond + denoise_guidance_scale * (noise_pred_text - noise_pred_uncond)
            denoised_latent = model.scheduler.step(noise_pred_guided, t, noisy_latent)["prev_sample"]
            after_img = ptp_utils.latent2image(model.vae, denoised_latent)[0]

        orig_labeled = ptp_utils.text_under_image(image.astype(np.uint8), "original")
        before_labeled = ptp_utils.text_under_image(before_img, f"before {label_t}")
        after_labeled = ptp_utils.text_under_image(after_img, f"after {label_t}")
        rows.append([orig_labeled, before_labeled, after_labeled])

    flat = [img for triple in rows for img in triple]
    ptp_utils.view_images(flat, num_rows=num_vis_steps, offset_ratio=0.01)


def aggregate_attention(
    attention_store: AttentionStore,
    tokenizer,
    prompts: List[str],
    res: int,
    from_where: List[str],
    is_cross: bool,
    select: int,
):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(
    attention_store: AttentionStore,
    tokenizer,
    prompts: List[str],
    res: int,
    from_where: List[str],
    select: int = 0,
):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, tokenizer, prompts, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))


def show_self_attention_comp(
    attention_store: AttentionStore,
    tokenizer,
    prompts: List[str],
    res: int,
    from_where: List[str],
    max_com: int = 10,
    select: int = 0,
):
    attention_maps = (
        aggregate_attention(attention_store, tokenizer, prompts, res, from_where, False, select)
        .numpy()
        .reshape((res**2, res**2))
    )
    _u, _s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))
