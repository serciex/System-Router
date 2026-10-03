"""Single-token answer labels for System 1, validated once against the tokenizer.

Same method as sgoedecke/system-one (`system_one/inference.py` index discovery and `demo/labels.py`
two-letter labels): an option label is only usable if it is one non-special token and decodes exactly
after the answer prefix. Numeric indexes cover about 10 options on Qwen tokenizers; two-letter labels
cover up to 100.
"""

from __future__ import annotations

import itertools
import string


class OptionTokens:
    def __init__(self, tokenizer, prefix: str, max_options: int = 100):
        self.prefix = prefix
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        decoded_prefix = tokenizer.decode(prefix_ids, clean_up_tokenization_spaces=False)
        special = set(tokenizer.all_special_ids)

        def single(text: str):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1 or ids[0] in special:
                return None
            if tokenizer.decode(prefix_ids + ids, clean_up_tokenization_spaces=False) != decoded_prefix + text:
                return None
            return ids[0]

        self.index: list[tuple[str, int]] = []
        for i in range(max_options):
            token = single(str(i))
            if token is None:
                break
            self.index.append((str(i), token))

        self.letters: list[tuple[str, int]] = []
        seen: set[int] = set()
        for pair in itertools.product(string.ascii_uppercase, repeat=2):
            label = "".join(pair)
            token = single(label)
            if token is None or token in seen:
                continue
            seen.add(token)
            self.letters.append((label, token))
            if len(self.letters) == max_options:
                break
        if not self.index and not self.letters:
            raise ValueError("Tokenizer has no usable single-token option labels")

    def labels(self, n: int, mode: str = "auto") -> list[tuple[str, int]]:
        if mode == "index" or (mode == "auto" and n <= len(self.index)):
            table = self.index
        else:
            table = self.letters
        if n > len(table):
            raise ValueError(f"{n} options but only {len(table)} single-token labels are available")
        return table[:n]
