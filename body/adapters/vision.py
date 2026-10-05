"""PENDING PORT to contract v0.3 (still uses the v0.2 adapter interface). Not used by the v3 loop.

Vision-only adapter: interactable elements detected from pixels (e.g. OmniParser).

Environment-independent: it needs a `detector(image) -> list[dict]` and a screen object with
`image`, `pointer`, `click_at(x, y)`, `type_text(text)`, `point(x, y)` and `scroll(direction)`
(MiniWoBEnvironment provides these). Everything it reports has confidence below 1.0, and handles are
not stable across frames, so the contract tracks IDs by position.

Detector output items: {"bbox": (x0, y0, x1, y1) normalized, "label": str, "interactable": bool,
"typeable": bool (optional), "score": float (optional)}.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..schema import Capabilities, Frame, NativeElement
from .base import Adapter, result

Detector = Callable[[object], list[dict]]


class VisionAdapter(Adapter):
    name = "vision"

    def __init__(self, screen, detector: Detector, confidence: float = 0.6):
        self.screen = screen
        self.detector = detector
        self.confidence = float(confidence)
        self._boxes: dict[int, tuple] = {}

    def capabilities(self) -> Capabilities:
        return Capabilities(pointer="absolute", verbs=("click", "type", "scroll_up", "scroll_down"),
                            stable_ids=False, realtime=False, has_source=False)

    def reset(self, seed: Optional[int] = None) -> Frame:
        self.screen.reset(seed=seed)
        return self.read()

    def read(self) -> Frame:
        image = self.screen.image
        elements, self._boxes = [], {}
        for index, item in enumerate(self.detector(image)):
            if not item.get("interactable", True):
                continue
            box = tuple(float(v) for v in item["bbox"])
            verbs = ("click", "type") if item.get("typeable") else ("click",)
            score = float(item.get("score", self.confidence))
            self._boxes[index] = box
            elements.append(NativeElement(handle=index, bbox=box, label=str(item.get("label") or "icon"),
                                          role="detected", native_actions=verbs,
                                          confidence=min(self.confidence, score)))
        return Frame(image=image, anchor=self.screen.pointer, elements=elements,
                     text=getattr(self.screen, "utterance", ""))

    def invoke(self, handle, native_action: str, arg: Optional[str] = None) -> dict:
        if native_action in ("scroll_up", "scroll_down"):
            return {**self.screen.scroll(native_action.split("_")[1]), "text": native_action.replace("_", " ")}
        box = self._boxes.get(handle)
        if box is None:
            return result(False, "detected element no longer present")
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if native_action == "click":
            return {**self.screen.click_at(x, y), "text": "clicked"}
        if native_action == "type":
            res = self.screen.click_at(x, y)
            if not res.get("ok") or res.get("done"):
                return res
            return {**self.screen.type_text(arg or ""), "text": f'typed "{arg or ""}"'}
        return result(False, f"unsupported verb {native_action}")

    def point(self, x: float, y: float) -> dict:
        return {**self.screen.point(x, y), "text": "pointer moved"}
