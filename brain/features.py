"""The world model's inputs, taken from the same frozen VLM.

Vision: the VLM's vision tower runs on the screenshot (no language pass), its merged patch features are
pooled to a fixed grid, and a fixed seeded random projection shrinks each cell to `cell_dim`.
Text: System 1's hidden state after the last outcome (see System1.begin_step), projected to `text_dim`.

The projections are fixed (never trained) so the world model always sees the same feature space.
"""

from __future__ import annotations

import zlib

import numpy as np

from .llm import LLM


class FeatureExtractor:
    def __init__(self, llm: LLM, vision_grid: int = 12, cell_dim: int = 32, text_dim: int = 256, seed: int = 0):
        self.llm = llm
        self.grid = int(vision_grid)
        self.cell_dim = int(cell_dim)
        self.text_dim = int(text_dim)
        self.seed = int(seed)
        self._projections: dict[tuple[str, int, int], np.ndarray] = {}

    @property
    def vision_size(self) -> int:
        return self.grid * self.grid * self.cell_dim

    def _projection(self, name: str, d_in: int, d_out: int) -> np.ndarray:
        key = (name, d_in, d_out)
        if key not in self._projections:
            rng = np.random.RandomState(self.seed + zlib.crc32(name.encode()) % 100000)
            self._projections[key] = (rng.standard_normal((d_in, d_out)) / np.sqrt(d_in)).astype(np.float32)
        return self._projections[key]

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        return ((x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)).astype(np.float32)

    def text(self, hidden: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)
        return self._normalize(hidden @ self._projection("text", hidden.shape[0], self.text_dim))

    def vision(self, image: np.ndarray) -> np.ndarray:
        if not self.llm.is_vlm:
            return np.zeros(self.vision_size, dtype=np.float32)
        import torch
        import torch.nn.functional as F
        from PIL import Image

        processor = self.llm.processor.image_processor
        inputs = processor(images=[Image.fromarray(image)], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.llm.device, dtype=self.llm.model.dtype)
        grid_thw = inputs["image_grid_thw"].to(self.llm.device)
        with torch.inference_mode():
            feats = _image_features(self.llm.model, pixel_values, grid_thw)
        merge = int(getattr(self.llm.model.config.vision_config, "spatial_merge_size", 2))
        _, h, w = (int(v) for v in grid_thw[0])
        rows, cols = h // merge, w // merge
        feats = feats[: rows * cols].float().reshape(rows, cols, -1).permute(2, 0, 1)[None]
        pooled = F.adaptive_avg_pool2d(feats, self.grid)[0].permute(1, 2, 0).reshape(self.grid * self.grid, -1)
        pooled = pooled.cpu().numpy()
        cells = self._normalize(pooled @ self._projection("vision", pooled.shape[1], self.cell_dim))
        return cells.reshape(-1)


def _image_features(model, pixel_values, grid_thw):
    """Merged vision-tower output, (num_tokens, hidden). Handles the API differences between versions."""
    if hasattr(model, "get_image_features"):
        out = model.get_image_features(pixel_values, image_grid_thw=grid_thw)
    else:
        out = model.model.visual(pixel_values, grid_thw=grid_thw)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        out = out.pooler_output
    elif hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.reshape(-1, out.shape[-1])
