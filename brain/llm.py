"""Load the one VLM (Qwen3.5) used for act, think and grounding.

Qwen3.5 is natively multimodal, so it loads through the image-text model class and its processor. A
text-only causal LM also works (without the screen). Needs a recent `transformers`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from common.config import Config, model_source

SENTINEL = "<<<SYSTEM_ROUTER_CONTENT>>>"


@dataclass
class LLM:
    model: Any
    tokenizer: Any
    processor: Any  # None for text-only models
    device: Any

    @property
    def is_vlm(self) -> bool:
        return self.processor is not None

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def chat_parts(self) -> tuple[str, str]:
        """Split the chat template around the user content, with thinking disabled.

        Returns (head, tail): head + content + tail is the full prompt up to the assistant's first token.
        """
        if self.tokenizer.chat_template:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": SENTINEL}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            head, tail = rendered.split(SENTINEL)
            return head, tail
        return "", "\nAssistant:\n"


def load_llm(cfg: Config) -> LLM:
    import torch
    import transformers

    source = model_source(cfg)
    dtype = getattr(torch, cfg.llm.dtype)
    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": cfg.llm.device_map}
    if cfg.llm.quantization in ("4bit", "8bit"):
        if cfg.llm.quantization == "4bit":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
        else:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)

    processor = None
    try:
        model = transformers.AutoModelForImageTextToText.from_pretrained(source, **kwargs)
        processor = transformers.AutoProcessor.from_pretrained(source)
        tokenizer = processor.tokenizer
    except (ValueError, KeyError):
        model = transformers.AutoModelForCausalLM.from_pretrained(source, **kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(source)
    model.eval()
    device = model.get_input_embeddings().weight.device
    return LLM(model=model, tokenizer=tokenizer, processor=processor, device=device)


def generate(llm: LLM, prompt: str, image=None, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
    """Free generation (System 2, labeler). Thinking is disabled; the prompt asks for JSON."""
    import torch
    from PIL import Image

    sample = temperature > 0
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": sample}
    if sample:
        gen_kwargs["temperature"] = temperature
    if image is not None and llm.is_vlm:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = llm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = llm.processor(text=[text], images=[Image.fromarray(image)], return_tensors="pt").to(llm.device)
    else:
        head, tail = llm.chat_parts()
        inputs = llm.tokenizer(head + prompt + tail, return_tensors="pt", add_special_tokens=False).to(llm.device)
    with torch.inference_mode():
        output = llm.model.generate(**inputs, **gen_kwargs)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return llm.tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_json(text: str) -> Optional[dict]:
    """First JSON object in the text, or None."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
