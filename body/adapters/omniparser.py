"""OmniParser detector for the vision adapter (weights added later).

Expects a checkout of https://github.com/microsoft/OmniParser on the Python path (its `util` package)
and its weights in `weights_dir` (`icon_detect/model.pt` and `icon_caption_florence`). The OmniParser
API changes between versions; this wrapper targets its `get_som_labeled_img` helper and normalizes the
output to the detector format in vision.py. Verify against the installed version before relying on it.
"""

from __future__ import annotations

from pathlib import Path


def load_omniparser(weights_dir: str | Path, box_threshold: float = 0.05):
    from PIL import Image
    from util.utils import check_ocr_box, get_caption_model_processor, get_som_labeled_img, get_yolo_model

    weights_dir = Path(weights_dir)
    yolo = get_yolo_model(model_path=str(weights_dir / "icon_detect" / "model.pt"))
    caption = get_caption_model_processor(model_name="florence2", model_name_or_path=str(weights_dir / "icon_caption_florence"))

    def detect(image) -> list[dict]:
        pil = Image.fromarray(image)
        (ocr_text, ocr_boxes), _ = check_ocr_box(pil, display_img=False, output_bb_format="xyxy",
                                                 easyocr_args={"paragraph": False, "text_threshold": 0.9},
                                                 use_paddleocr=False)
        _, _, parsed = get_som_labeled_img(pil, yolo, BOX_TRESHOLD=box_threshold, output_coord_in_ratio=True,
                                           ocr_bbox=ocr_boxes, caption_model_processor=caption, ocr_text=ocr_text,
                                           use_local_semantics=True, iou_threshold=0.7, scale_img=False)
        items = []
        for entry in parsed:
            x0, y0, x1, y1 = (float(v) for v in entry["bbox"])
            content = str(entry.get("content") or "").strip()
            items.append({
                "bbox": (max(x0, 0.0), max(y0, 0.0), min(x1, 1.0), min(y1, 1.0)),
                "label": content or entry.get("type", "icon"),
                "interactable": bool(entry.get("interactivity", True)) or entry.get("type") == "text",
                "typeable": "input" in content.lower() or "search" in content.lower(),
            })
        return items

    return detect
