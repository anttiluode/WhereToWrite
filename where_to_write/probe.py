"""Frozen Qwen microscope: identical value pulses at different token sites.

No model training, no J-lens computation. The default color direction is the
final gamma-scaled unembedding row (a *logit-lens* direction). This is deliberate:
Claude's G4.2 found J-lens and logit-lens both 100% at the answer token. A cached
J-lens may optionally be loaded to compare, but it is not required by the demo.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .memory import COLORS

PROMPTS = (
    'What color is {name}? Answer:',
    'Question: What is the color of {name}? Answer:',
    'Give only the color of {name}. Answer:',
)
ARMS = ('baseline', 'subject', 'answer', 'other_token', 'random_subject', 'wrong_subject')


def top_token_is_color(token: str, color: str) -> bool:
    """Recognize capitalization variants without changing exact-token strict accuracy."""
    return token.strip().casefold() == color.casefold()


def subject_span(text: str, name: str) -> tuple[int, int]:
    """Find the last whole-word name occurrence (without guessing token IDs)."""
    if not name.strip():
        raise ValueError('Empty name')
    pattern = re.compile(r'(?<!\w)' + re.escape(name.strip()) + r'(?!\w)', re.I)
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f'Name {name!r} absent from question')
    return matches[-1].span()


def overlapping_positions(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    """Tokenizer offsets for text span, robust to multi-token names."""
    first, end = span
    return [i for i, (a, b) in enumerate(offsets) if b > first and a < end]


@dataclass
class ArmResult:
    arm: str
    top_token: str
    strict_correct: bool
    choice: str
    choice_correct: bool
    conditional_color_probs: dict[str, float]
    true_margin: float
    semantic_top_correct: bool = False


class QwenProbe:
    def __init__(self, model_name='Qwen/Qwen2.5-0.5B-Instruct', layer=16, device='auto',
                 j_lens_cache: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model_name, self.layer = model_name, layer
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tok.is_fast:
            raise ValueError('Fast tokenizer needed for accurate token offsets')
        dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype,
                 attn_implementation='eager').to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.layers = self.model.model.layers
        if not 0 <= layer < len(self.layers):
            raise ValueError(f'Choose a layer from 0 to {len(self.layers) - 1}')
        self.color_ids = [self._single(' ' + color) for color in COLORS]
        self._jcache = None
        if j_lens_cache:
            # Only load locally generated, trusted research checkpoints.
            checkpoint = torch.load(j_lens_cache, map_location='cpu', weights_only=True)
            self._jcache = checkpoint['J'][layer].float().to(self.device)
        self.direction_kind = 'J-lens' if self._jcache is not None else 'logit-lens'
        self.directions = self._directions()
        g = torch.Generator(device='cpu').manual_seed(20261018)
        self.random_direction = torch.nn.functional.normalize(
            torch.randn(self.directions.shape[1], generator=g), dim=0).to(self.device)

    def _single(self, text: str) -> int:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'Expected one token for {text!r}, got {len(ids)}')
        return ids[0]

    def _directions(self):
        torch = self.torch
        W = self.model.get_output_embeddings().weight.detach().float()[self.color_ids]
        gamma = self.model.model.norm.weight.detach().float()
        v = W * gamma.unsqueeze(0)
        if self._jcache is not None:
            v = v @ self._jcache  # row convention: v = (gamma * u) @ J
        return torch.nn.functional.normalize(v, dim=-1)

    def _positions(self, text: str, name: str):
        enc = self.tok(text, return_tensors='pt', return_offsets_mapping=True, add_special_tokens=False)
        offsets = [tuple(map(int, pair)) for pair in enc.pop('offset_mapping')[0].tolist()]
        name_positions = overlapping_positions(offsets, subject_span(text, name))
        color_match = re.search(r'\bcolor\b', text, flags=re.I)
        other_positions = overlapping_positions(offsets, color_match.span()) if color_match else []
        if not name_positions or not other_positions:
            raise ValueError('Could not resolve subject or control token positions')
        # The last name subtoken carries the site; multi-token names are not silently rejected.
        return {k: v.to(self.device) for k, v in enc.items()}, name_positions[-1], other_positions[-1]

    def _logits(self, encoded, arm: str, color: str, subject_pos: int, other_pos: int, gain: float):
        torch = self.torch
        if arm == 'baseline':
            with torch.inference_mode():
                return self.model(**encoded, use_cache=False).logits[0, -1].float().cpu()
        target = COLORS.index(color)
        site = subject_pos if arm in ('subject', 'random_subject', 'wrong_subject') else (
            other_pos if arm == 'other_token' else encoded['input_ids'].shape[1] - 1)
        if arm == 'wrong_subject':
            direction = self.directions[(target + 1) % len(COLORS)]
        elif arm == 'random_subject':
            direction = self.random_direction
        else:
            direction = self.directions[target]

        def hook(_module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            changed = hidden.clone()
            pulse = direction.to(changed.dtype) * (float(gain) * hidden[0, site].float().norm())
            changed[0, site] += pulse
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handle = self.layers[self.layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                return self.model(**encoded, use_cache=False).logits[0, -1].float().cpu()
        finally:
            handle.remove()

    def _logits_many(self, encoded, arms, color: str, subject_pos: int, other_pos: int,
                     gain: float):
        """Matched interventions in one batch: rows cannot attend to other rows.

        Reduces launch overhead during long GPU benchmarks. Equivalent to
        independent forwards in eval mode up to batch-kernel roundoff.
        """
        torch = self.torch
        n = len(arms)
        batched = {key: value.expand(n, *value.shape[1:]).contiguous()
                   for key, value in encoded.items()}
        t_len = encoded['input_ids'].shape[1]
        target = COLORS.index(color)

        def hook(_module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            changed = hidden.clone()
            for i, arm in enumerate(arms):
                if arm == 'baseline':
                    continue
                site = (subject_pos if arm in ('subject', 'random_subject', 'wrong_subject')
                        else other_pos if arm == 'other_token' else t_len - 1)
                if arm == 'wrong_subject':
                    direction = self.directions[(target + 1) % len(COLORS)]
                elif arm == 'random_subject':
                    direction = self.random_direction
                else:
                    direction = self.directions[target]
                pulse = direction.to(changed.dtype) * (float(gain) * hidden[i, site].float().norm())
                changed[i, site] += pulse
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handle = self.layers[self.layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                logits = self.model(**batched, use_cache=False).logits[:, -1].float().cpu()
        finally:
            handle.remove()
        return {arm: logits[i] for i, arm in enumerate(arms)}

    def evaluate(self, name: str, color: str, template: str = PROMPTS[0],
                 gain: float = 0.5, arms=ARMS) -> dict[str, ArmResult]:
        if color not in COLORS or '{name}' not in template:
            raise ValueError('Use a supported color and a prompt template containing {name}')
        if not 0 <= gain <= 8:
            raise ValueError('Gain must lie between 0 and 8')
        text = template.format(name=name)
        encoded, sp, cp = self._positions(text, name)
        torch = self.torch
        output = {}
        target = COLORS.index(color)
        if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
            raise ValueError('Arms must be a nonempty set of known unique interventions')
        logits_by_arm = self._logits_many(encoded, arms, color, sp, cp, gain)
        for arm in arms:
            logits = logits_by_arm[arm]
            candidate_logits = logits[self.color_ids]
            probs = torch.softmax(candidate_logits, dim=0)
            predicted = int(candidate_logits.argmax())
            other_max = torch.cat([candidate_logits[:target], candidate_logits[target+1:]]).max()
            best_token = int(logits.argmax())
            output[arm] = ArmResult(
                arm, self.tok.decode([best_token]), best_token == self.color_ids[target],
                COLORS[predicted], predicted == target,
                {c: float(probs[i]) for i, c in enumerate(COLORS)},
                float(candidate_logits[target] - other_max),
                top_token_is_color(self.tok.decode([best_token]), color))
        return output
