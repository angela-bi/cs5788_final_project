"""
CLIP directional similarity for text-guided image edits (no ground-truth edited image).

Computes cosine similarity between
    Δ_image = CLIP_embed(I_edited) - CLIP_embed(I_original)
    Δ_text  = CLIP_embed(P_target) - CLIP_embed(P_source)
using L2-normalized CLIP image/text features (ViT-B/32 by default), matching the common
definition: "did the image change in the same direction in embedding space as the prompt?"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_ImageLike = Union[np.ndarray, torch.Tensor, Image.Image]


def _to_pil_rgb(image: _ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0)
        if float(arr.max()) <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


class CLIPDirectionalMetrics:
    """Lazy-loads CLIP once; safe to reuse across many edits on the same kernel."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_id = model_id
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(self._device)
        self._model.eval()

    @torch.no_grad()
    def _encode_image(self, image: _ImageLike) -> torch.Tensor:
        self._lazy_load()
        pil = _to_pil_rgb(image)
        inputs = self._processor(images=[pil], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        feats = self._model.get_image_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        self._lazy_load()
        inputs = self._processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        feats = self._model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def evaluate_edit(
        self,
        original: _ImageLike,
        edited: _ImageLike,
        source_prompt: str,
        target_prompt: str,
        reconstruction: Optional[_ImageLike] = None,
    ) -> Dict[str, float]:
        z_img_o = self._encode_image(original)
        z_img_e = self._encode_image(edited)
        z_txt = self._encode_text_batch([source_prompt, target_prompt])
        z_txt_s, z_txt_t = z_txt[0:1], z_txt[1:2]

        d_img = z_img_e - z_img_o
        d_txt = z_txt_t - z_txt_s
        directional = F.cosine_similarity(d_img, d_txt, dim=-1).item()

        clip_edited_target = F.cosine_similarity(z_img_e, z_txt_t, dim=-1).item()
        clip_original_source = F.cosine_similarity(z_img_o, z_txt_s, dim=-1).item()

        out: Dict[str, float] = {
            "directional_clip": float(directional),
            "clip_image_text_edited_target": float(clip_edited_target),
            "clip_image_text_original_source": float(clip_original_source),
        }
        if reconstruction is not None:
            z_img_r = self._encode_image(reconstruction)
            out["clip_image_text_reconstruction_source"] = float(
                F.cosine_similarity(z_img_r, z_txt_s, dim=-1).item()
            )
            out["clip_image_image_reconstruction_original"] = float(
                F.cosine_similarity(z_img_r, z_img_o, dim=-1).item()
            )
        return out

    def evaluate_result_dict(self, result: Dict[str, Any], name: str = "") -> Dict[str, Any]:
        """Use keys from `edit_real_image` / `edit_real_image_null` returns."""
        missing = [k for k in ("original", "edited", "source_prompt", "target_prompt") if k not in result]
        if missing:
            raise KeyError(f"result dict missing keys {missing}")
        metrics = self.evaluate_edit(
            result["original"],
            result["edited"],
            str(result["source_prompt"]),
            str(result["target_prompt"]),
            reconstruction=result.get("reconstruction"),
        )
        row: Dict[str, Any] = {"name": name or "edit", **metrics}
        row["source_prompt"] = result["source_prompt"]
        row["target_prompt"] = result["target_prompt"]
        return row


def evaluate_many(
    clip_metrics: CLIPDirectionalMetrics,
    cases: List[tuple[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, res in cases:
        rows.append(clip_metrics.evaluate_result_dict(res, name=name))
    return rows


def rows_to_markdown_table(rows: List[Dict[str, Any]], float_decimals: int = 3) -> str:
    """Compact table for papers / notebook Markdown."""
    if not rows:
        return "_No rows._"
    skip = {"name", "source_prompt", "target_prompt"}
    numeric_keys = sorted(
        {
            k
            for r in rows
            for k, v in r.items()
            if k not in skip and isinstance(v, (int, float, np.floating))
        }
    )
    headers = ["name", *numeric_keys]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        cells = [str(r.get("name", ""))]
        for k in numeric_keys:
            v = r.get(k, "")
            if isinstance(v, float):
                cells.append(f"{v:.{float_decimals}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
