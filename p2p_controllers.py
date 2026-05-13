# Prompt-to-Prompt attention controllers (Stable Diffusion) adapted from the
# original prompt-to-prompt notebook code. Requires `init_project()` in
# `real_image_edit` (or `set_runtime` below) before constructing controllers.

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as nnf

import ptp_utils
from seq_aligner import get_refinement_mapper, get_replacement_mapper, get_word_inds

_RUNTIME: Optional["P2PRuntime"] = None


class P2PRuntime:
    __slots__ = ("model", "device", "tokenizer", "num_ddim_steps", "guidance_scale", "max_num_words")

    def __init__(
        self,
        model,
        num_ddim_steps: int = 50,
        guidance_scale: float = 7.5,
        max_num_words: int = 77,
    ) -> None:
        self.model = model
        self.device = model.device
        self.tokenizer = model.tokenizer
        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale
        self.max_num_words = max_num_words


def set_runtime(rt: P2PRuntime) -> None:
    global _RUNTIME
    _RUNTIME = rt


def get_runtime() -> P2PRuntime:
    if _RUNTIME is None:
        raise RuntimeError("Call real_image_edit.init_project(model) before using P2P controllers.")
    return _RUNTIME


class LocalBlend:
    def __call__(self, x_t, attention_store, step):
        k = 1
        maps = attention_store["down_cross"][:2] + attention_store["up_cross"][3:6]
        rt = get_runtime()
        maps = [
            item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, rt.max_num_words) for item in maps
        ]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: Union[List[List[str]], List[str], str], threshold: float = 0.3):
        rt = get_runtime()
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, rt.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, rt.tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(rt.device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        h = attn.shape[0]
        attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if attn.shape[1] <= 16**2:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn.detach())
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        return {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16**2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                    1 - alpha_words
                ) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
    ):
        super(AttentionControlEdit, self).__init__()
        rt = get_runtime()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, rt.tokenizer, max_num_words=rt.max_num_words
        ).to(rt.device)
        if isinstance(self_replace_steps, float):
            self_replace_steps = (0.0, self_replace_steps)
        self.num_self_replace = (
            int(num_steps * self_replace_steps[0]),
            int(num_steps * self_replace_steps[1]),
        )
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
    ):
        super(AttentionReplace, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend
        )
        rt = get_runtime()
        self.mapper = get_replacement_mapper(prompts, rt.tokenizer).to(rt.device)


class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
    ):
        super(AttentionRefine, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend
        )
        rt = get_runtime()
        self.mapper, alphas = get_refinement_mapper(prompts, rt.tokenizer)
        self.mapper, alphas = self.mapper.to(rt.device), alphas.to(rt.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        equalizer,
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[AttentionControlEdit] = None,
    ):
        super(AttentionReweight, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend
        )
        rt = get_runtime()
        self.equalizer = equalizer.to(rt.device)
        self.prev_controller = controller


def get_equalizer(
    text: str,
    word_select: Union[int, Tuple[int, ...], str],
    values: Union[List[float], Tuple[float, ...]],
):
    rt = get_runtime()
    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = get_word_inds(text, word, rt.tokenizer)
        for i in inds:
            equalizer[:, i] = values
    return equalizer


def make_controller(
    prompts: List[str],
    is_replace_controller: bool,
    cross_replace_steps: Dict[str, float],
    self_replace_steps: float,
    blend_words=None,
    equilizer_params=None,
) -> AttentionControlEdit:
    """Build a stacked P2P controller (optional local blend + optional reweight)."""
    rt = get_runtime()
    num_steps = rt.num_ddim_steps
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words)
    if is_replace_controller:
        controller = AttentionReplace(
            prompts, num_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb
        )
    else:
        controller = AttentionRefine(
            prompts, num_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb
        )
    if equilizer_params is not None:
        eq = get_equalizer(
            prompts[1], equilizer_params["words"], equilizer_params["values"]
        )
        controller = AttentionReweight(
            prompts,
            num_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            equalizer=eq,
            local_blend=lb,
            controller=controller,
        )
    return controller
