"""
Minimal smoke test for Prompt-to-Prompt attention hooking.

This is intentionally small and does not try to reproduce the full paper results.
It verifies the most failure-prone parts of getting the notebook working:
1) the Hugging Face model can be downloaded/loaded
2) `register_attention_control` finds CrossAttention layers and patches them
3) the controller is invoked and stores attention maps during a (very short) run
"""

from __future__ import annotations

import abc
from typing import Dict, List

import ptp_utils
import torch
from diffusers import DiffusionPipeline


class AttentionControl(abc.ABC):
    """Base controller: orchestrates step/layer bookkeeping and CFG split.

    Important behavior:
    - The model runs classifier-free guidance (CFG) by concatenating unconditional and conditional
      batches. In `ptp_utils.register_attention_control`, attention is shaped as (batch*heads, ...).
      The first half corresponds to unconditional, the second half to conditional.
    - Prompt-to-Prompt edits should affect only the conditional pass, so `__call__` sends only the
      conditional slice into `forward`.
    """

    def step_callback(self, x_t: torch.Tensor) -> torch.Tensor:
        return x_t

    def between_steps(self) -> None:
        return

    @abc.abstractmethod
    def forward(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        # attn is (batch*heads, query_len, key_len)
        h = attn.shape[0]
        attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)

        # Bookkeeping: used by more advanced controllers (replace/refine/reweight).
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self) -> None:
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self) -> None:
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    """Stores attention maps so we can confirm the hook fires."""

    @staticmethod
    def get_empty_store() -> Dict[str, List[torch.Tensor]]:
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        # Only store moderate-resolution maps to keep memory manageable.
        # For LDM 256x256, 16x16 attention corresponds to 256 queries.
        if attn.shape[1] <= 16**2:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn.detach())
        return attn

    def between_steps(self) -> None:
        # Aggregate per-step stores.
        if not self.attention_store:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self) -> Dict[str, List[torch.Tensor]]:
        # Avoid divide by zero in case something calls this too early.
        denom = max(self.cur_step, 1)
        return {k: [x / denom for x in v] for k, v in self.attention_store.items()}

    def reset(self) -> None:
        super().reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self) -> None:
        super().__init__()
        self.step_store = self.get_empty_store()
        self.attention_store: Dict[str, List[torch.Tensor]] = {}


def main() -> None:
    pipe = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256").to("cpu")

    controller = AttentionStore()
    ptp_utils.register_attention_control(pipe, controller)

    if controller.num_att_layers <= 0:
        raise RuntimeError(
            f"register_attention_control did not find CrossAttention layers. num_att_layers={controller.num_att_layers}"
        )

    # Very short run (1 step) just to verify the controller is called.
    _images, _latent = ptp_utils.text2image_ldm(
        pipe,
        prompt=["a red apple"],
        controller=controller,
        num_inference_steps=1,
        guidance_scale=5.0,
        generator=torch.Generator().manual_seed(0),
    )

    avg = controller.get_average_attention()
    counts = {k: len(v) for k, v in avg.items()}
    print("OK: attention store counts:", counts)
    print("OK: num_att_layers:", controller.num_att_layers)


if __name__ == "__main__":
    main()

