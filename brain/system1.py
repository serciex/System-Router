"""Act mode: constrained single-pass picks on one cached multimodal prompt (the system-one method).

The prompt (with the screen image) is prefilled once per step. Each question then runs one short pass
on a copy of that cache, with logits restricted to its own option labels. Labels are two-letter codes,
unique across both questions. If the model's cache cannot be copied, set `reuse_cache=False` and each
question recomputes the full prompt (same answers, slower). Qwen3.5's hybrid attention must be checked.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from body.prompt import Labeled, PromptParts, question_text
from body.schema import Option

from .labels import OptionTokens
from .llm import LLM, SENTINEL

PREFIX = "choice:"
ANSWER_RULE = f"Answer with {PREFIX} followed immediately by one option label."


@dataclass
class Answer:
    key: str
    label: str
    probs: dict[str, float]
    top_prob: float
    confidence: float  # 1 - normalized entropy


class System1:
    def __init__(self, llm: LLM, max_options: int = 100, reuse_cache: bool = True):
        self.llm = llm
        self.tokens = OptionTokens(llm.tokenizer, PREFIX, max_options * 2)
        self.max_options = max_options
        self.reuse_cache = reuse_cache
        self._inputs: dict[str, Any] = {}
        self._cache: Any = None
        self._tail = ""

    # ------------------------------------------------------------------ labels
    def labels(self, navigation: dict[str, list[Option]], action: list[Option]) -> dict[str, Labeled]:
        """Unique two-letter labels across every question, in prompt order."""
        questions = {**{k: v[: self.max_options] for k, v in navigation.items()}, "action": action[: self.max_options]}
        table = self.tokens.labels(sum(len(v) for v in questions.values()), "letters")
        out, i = {}, 0
        for name, options in questions.items():
            out[name] = [(table[i + j][0], table[i + j][1], option) for j, option in enumerate(options)]
            i += len(options)
        return out

    # ------------------------------------------------------------------ prefill
    def _render(self, parts: PromptParts) -> tuple[str, str]:
        after = parts.after_image() + f"\n{ANSWER_RULE}\n" + SENTINEL
        if self.llm.is_vlm and parts.images:
            content = [{"type": "text", "text": parts.before_image()}]
            content += [{"type": "image"} for _ in parts.images]
            content += [{"type": "text", "text": after}]
            rendered = self.llm.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                              add_generation_prompt=True, enable_thinking=False)
        else:
            head, tail = self.llm.chat_parts()
            rendered = head + parts.before_image() + "[screen not available]" + after + tail
        prefix, tail = rendered.split(SENTINEL)
        return prefix, tail

    def _encode_prefix(self, prefix: str, parts: PromptParts) -> dict[str, Any]:
        from PIL import Image

        if self.llm.is_vlm and parts.images:
            images = [Image.fromarray(image) for image in parts.images]
            return dict(self.llm.processor(text=[prefix], images=images, return_tensors="pt").to(self.llm.device))
        return dict(self.llm.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.llm.device))

    def prepare(self, parts: PromptParts) -> None:
        """Prefill the step's prompt once (image included)."""
        import torch

        prefix, self._tail = self._render(parts)
        self._inputs = self._encode_prefix(prefix, parts)
        self._cache = None
        if self.reuse_cache:
            with torch.inference_mode():
                self._cache = self.llm.model(**self._inputs, use_cache=True, return_dict=True).past_key_values

    def _logits(self, suffix: str):
        import torch

        ids = torch.tensor([self.llm.encode(suffix)], device=self.llm.device)
        with torch.inference_mode():
            if self.reuse_cache and self._cache is not None:
                out = self.llm.model(input_ids=ids, past_key_values=copy.deepcopy(self._cache), use_cache=True,
                                     return_dict=True)
            else:
                inputs = dict(self._inputs)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], dim=1)
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                out = self.llm.model(**inputs, return_dict=True)
        return out.logits[0, -1].float()

    # ------------------------------------------------------------------ questions
    def _choose(self, suffix: str, labeled: Labeled) -> Answer:
        import torch

        if len(labeled) == 1:
            label, _, option = labeled[0]
            return Answer(option.key, label, {option.key: 1.0}, 1.0, 1.0)
        logits = self._logits(suffix)
        allowed = torch.tensor([token for _, token, _ in labeled], device=logits.device)
        probs = torch.softmax(logits[allowed], dim=-1).cpu().tolist()
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        best = max(range(len(probs)), key=probs.__getitem__)
        label, _, option = labeled[best]
        return Answer(option.key, label, {o.key: p for (_, _, o), p in zip(labeled, probs)}, probs[best],
                      max(0.0, min(1.0, 1 - entropy / math.log(len(probs)))))

    def ask(self, name: str, labeled: Labeled) -> Answer:
        """One question whose options are already listed in the prompt."""
        return self._choose(f"Question: {question_text(name)}\n" + self._tail + PREFIX, labeled)

    def answer_all(self, labeled: dict[str, Labeled]) -> dict[str, Answer]:
        return {name: self.ask(name, entries) for name, entries in labeled.items()}

    def ask_extra(self, question: str, options: list[Option]) -> Answer:
        """A question whose options are not in the prompt (narrowing), listed in the suffix."""
        table = self.tokens.labels(len(options), "letters")
        labeled = [(label, token, option) for (label, token), option in zip(table, options)]
        listing = "\n".join(f"  {label}: {option.text}" for label, _, option in labeled)
        return self._choose(f"Question: {question}\nOptions:\n{listing}\n" + self._tail + PREFIX, labeled)
