from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from groundingdino.util import box_ops
from groundingdino.util.inference import load_image, load_model, predict
from huggingface_hub import hf_hub_download

try:  # Optional dependency; imported lazily when needed
    import open_clip
except Exception:  # pragma: no cover - optional dependency error path
    open_clip = None  # type: ignore

from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

GROUNDING_REPO = "ShilongLiu/GroundingDINO"
GROUNDING_WEIGHTS = "groundingdino_swint_ogc.pth"
GROUNDING_CONFIG = "GroundingDINO_SwinT_OGC.cfg.py"
SAM2_REPO = "facebook/sam2-hiera-large"


@dataclass
class AutoLabelerConfig:
    """Configuration for :class:`AutoLabeler`."""

    prompt: str = "defect . damage . flaw ."
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    top_k: int = 25
    min_area_ratio: float = 5e-4
    multimask_output: bool = False
    device: Optional[str] = None
    models_dir: Path = Path("~/.autodetector")
    support_manifest: Optional[Path] = None
    support_limit: Optional[int] = 16
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_threshold: float = 0.22
    overlay_alpha: float = 0.45

    def resolve(self) -> "AutoLabelerConfig":
        self.models_dir = self.models_dir.expanduser().resolve()
        if self.support_manifest is not None:
            self.support_manifest = self.support_manifest.expanduser().resolve()
        return self


@dataclass
class SupportExample:
    embedding: torch.Tensor
    label: str


@dataclass
class AutoLabelInstance:
    index: int
    score: float
    phrase: str
    bbox: List[int]
    area_ratio: float
    overlay_path: Path
    mask_path: Path
    crop_path: Path
    metadata_path: Path
    clip_score: Optional[float] = None


class AutoLabeler:
    """High-level pipeline that combines GroundingDINO and SAM 2 for pseudo labels."""

    def __init__(self, config: Optional[AutoLabelerConfig] = None) -> None:
        self.config = (config or AutoLabelerConfig()).resolve()
        self.device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        self._dino_model = self._load_groundingdino()
        self._sam_predictor = self._load_sam2()
        self._clip_model = None
        self._clip_preprocess = None
        self._support_examples: list[SupportExample] = []

        if self.config.support_manifest is not None:
            self._load_support_embeddings()

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _load_groundingdino(self):
        grounding_dir = self.config.models_dir / "groundingdino"
        grounding_dir.mkdir(parents=True, exist_ok=True)
        config_path = hf_hub_download(
            GROUNDING_REPO, GROUNDING_CONFIG, local_dir=grounding_dir
        )
        weights_path = hf_hub_download(
            GROUNDING_REPO, GROUNDING_WEIGHTS, local_dir=grounding_dir
        )
        model = load_model(str(config_path), str(weights_path))
        model.to(self.device)
        model.eval()
        return model

    def _load_sam2(self) -> SAM2ImagePredictor:
        sam2_dir = self.config.models_dir / "sam2"
        sam2_dir.mkdir(parents=True, exist_ok=True)
        config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[SAM2_REPO]
        checkpoint_path = hf_hub_download(
            SAM2_REPO, filename=checkpoint_name, local_dir=sam2_dir
        )
        sam_model = build_sam2(
            config_file=config_name, ckpt_path=str(checkpoint_path), device=str(self.device)
        )
        return SAM2ImagePredictor(sam_model)

    def _ensure_clip(self) -> None:
        if self._clip_model is not None:
            return
        if open_clip is None:
            raise RuntimeError(
                "open_clip is not installed. Install open-clip-torch to enable support filtering."
            )
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model,
            pretrained=self.config.clip_pretrained,
            device=self.device,
        )
        model.eval()
        self._clip_model = model
        self._clip_preprocess = preprocess

    def _load_support_embeddings(self) -> None:
        self._ensure_clip()
        assert self._clip_model is not None
        assert self._clip_preprocess is not None

        manifest_path = self.config.support_manifest
        if manifest_path is None:
            return
        with manifest_path.open("r", encoding="utf-8") as fp:
            manifest = json.load(fp)
        if not isinstance(manifest, Iterable):
            raise ValueError("Support manifest must be a list of entries.")

        examples: list[SupportExample] = []
        for entry in manifest:
            image_path = Path(entry["image"]).expanduser().resolve()
            bbox = entry.get("bbox")
            label = entry.get("label", "support")
            if bbox is None or len(bbox) != 4:
                raise ValueError("Each support entry must contain a bbox with four integers.")
            with Image.open(image_path) as pil_image:  # type: ignore[arg-type]
                crop = pil_image.convert("RGB").crop(tuple(bbox))
            tensor = self._clip_preprocess(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self._clip_model.encode_image(tensor)
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
            examples.append(SupportExample(embedding=embedding, label=label))
            if self.config.support_limit and len(examples) >= self.config.support_limit:
                break
        self._support_examples = examples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_directory(
        self,
        image_dir: Path,
        output_dir: Path,
        image_extensions: Optional[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp"),
        progress_callback: Callable[[int, int, Path], None] | None = None,
    ) -> list[list[AutoLabelInstance]]:
        image_dir = image_dir.expanduser().resolve()
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest: list[list[AutoLabelInstance]] = []
        image_paths = [
            image_path
            for image_path in sorted(image_dir.iterdir())
            if image_path.is_file()
        ]
        if image_extensions:
            image_paths = [
                image_path
                for image_path in image_paths
                if image_path.suffix.lower() in image_extensions
            ]

        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            if progress_callback is not None:
                progress_callback(index, total, image_path)
            instances = self.process_image(image_path, output_dir / image_path.stem)
            manifest.append(instances)
        return manifest

    def process_image(self, image_path: Path, output_dir: Path) -> list[AutoLabelInstance]:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_source, image_tensor = load_image(str(image_path))
        height, width = image_source.shape[:2]

        boxes, logits, phrases = predict(
            model=self._dino_model,
            image=image_tensor,
            caption=self.config.prompt,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            device=self.device,
        )
        if boxes.shape[0] == 0:
            return []

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes_xyxy *= torch.tensor([width, height, width, height], device=boxes_xyxy.device)
        boxes_xyxy = boxes_xyxy.cpu().numpy()
        logits = logits.sigmoid().cpu().numpy()

        self._sam_predictor.set_image(image_source)

        candidates = []
        for box_xyxy, logit_vec, phrase in zip(boxes_xyxy, logits, phrases):
            masks, iou_predictions, _ = self._sam_predictor.predict(
                box=box_xyxy,
                multimask_output=self.config.multimask_output,
                normalize_coords=True,
            )
            if self.config.multimask_output and masks.shape[0] > 1:
                best_idx = int(np.argmax(iou_predictions))
                mask = masks[best_idx]
            else:
                mask = masks[0]
            mask_bool = mask.astype(bool)
            area_ratio = float(mask_bool.sum()) / float(height * width)
            if area_ratio < self.config.min_area_ratio:
                continue
            candidates.append((box_xyxy, mask_bool, logit_vec, phrase, area_ratio))

        if not candidates:
            return []

        instances: list[tuple] = []
        for idx, (box_xyxy, mask_bool, logit_vec, phrase, area_ratio) in enumerate(candidates):
            bbox = self._sanitize_bbox(box_xyxy, width, height)
            if bbox is None:
                continue
            score = float(np.max(logit_vec))
            clip_score = self._clip_score(image_source, bbox) if self._support_examples else None
            ranking_score = clip_score if clip_score is not None else score
            instances.append((ranking_score, score, clip_score, bbox, mask_bool, phrase, area_ratio))

        if not instances:
            return []

        instances.sort(key=lambda item: item[0], reverse=True)
        if self.config.top_k:
            instances = instances[: self.config.top_k]

        created_instances: list[AutoLabelInstance] = []
        for local_index, instance in enumerate(instances):
            ranking_score, score, clip_score, bbox, mask_bool, phrase, area_ratio = instance
            if clip_score is not None and clip_score < self.config.clip_threshold:
                continue
            instance_dir = output_dir / f"instance_{local_index:02d}"
            instance_dir.mkdir(parents=True, exist_ok=True)

            overlay_path = instance_dir / "overlay.png"
            mask_path = instance_dir / "mask.png"
            crop_path = instance_dir / "bbox.png"
            metadata_path = instance_dir / "report.json"

            overlay_image = self._draw_overlay(
                image_source,
                np.asarray([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=np.float32),
                [mask_bool.astype(np.uint8)],
            )
            cv2.imwrite(str(overlay_path), overlay_image)

            self._save_mask(mask_bool, mask_path)
            crop_image = image_source[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            if crop_image.size == 0:
                crop_image = image_source[max(0, bbox[1]) : max(0, bbox[1] + 1), max(0, bbox[0]) : max(0, bbox[0] + 1)]
            crop_bgr = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_path), crop_bgr)

            metadata = {
                "score": score,
                "clip_score": clip_score,
                "bbox": bbox,
                "phrase": phrase,
                "area_ratio": area_ratio,
                "mask": str(mask_path.relative_to(output_dir.parent)),
                "overlay": str(overlay_path.relative_to(output_dir.parent)),
                "crop": str(crop_path.relative_to(output_dir.parent)),
            }
            with metadata_path.open("w", encoding="utf-8") as fp:
                json.dump(metadata, fp, ensure_ascii=False, indent=2)

            created_instances.append(
                AutoLabelInstance(
                    index=local_index,
                    score=score,
                    phrase=phrase,
                    bbox=bbox,
                    area_ratio=area_ratio,
                    overlay_path=overlay_path,
                    mask_path=mask_path,
                    crop_path=crop_path,
                    metadata_path=metadata_path,
                    clip_score=clip_score,
                )
            )

        manifest_path = output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fp:
            json.dump([self._instance_to_dict(instance) for instance in created_instances], fp, indent=2)

        return created_instances

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _clip_score(self, image_source: np.ndarray, bbox: List[int]) -> Optional[float]:
        if not self._support_examples:
            return None
        self._ensure_clip()
        assert self._clip_model is not None
        assert self._clip_preprocess is not None

        crop = image_source[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        if crop.size == 0:
            return None
        crop_pil = Image.fromarray(crop)
        tensor = self._clip_preprocess(crop_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self._clip_model.encode_image(tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        scores = [torch.matmul(embedding, example.embedding.T) for example in self._support_examples]
        stacked = torch.cat(scores, dim=-1)
        max_score = float(stacked.max().item())
        return max_score

    @staticmethod
    def _instance_to_dict(instance: AutoLabelInstance) -> dict:
        return {
            "index": instance.index,
            "score": instance.score,
            "clip_score": instance.clip_score,
            "phrase": instance.phrase,
            "bbox": instance.bbox,
            "area_ratio": instance.area_ratio,
            "overlay": instance.overlay_path.name,
            "mask": instance.mask_path.name,
            "crop": instance.crop_path.name,
            "metadata": instance.metadata_path.name,
        }

    @staticmethod
    def _sanitize_bbox(box_xyxy: np.ndarray, width: int, height: int) -> Optional[List[int]]:
        x0 = int(np.clip(round(float(box_xyxy[0])), 0, width - 1))
        y0 = int(np.clip(round(float(box_xyxy[1])), 0, height - 1))
        x1 = int(np.clip(round(float(box_xyxy[2])), x0 + 1, width))
        y1 = int(np.clip(round(float(box_xyxy[3])), y0 + 1, height))
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    def _save_mask(self, mask: np.ndarray, path: Path) -> None:
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 220
        rgba[..., 1] = 20
        rgba[..., 2] = 60
        rgba[..., 3] = np.where(mask, int(self.config.overlay_alpha * 255), 0)
        bgra = rgba[..., [2, 1, 0, 3]]
        cv2.imwrite(str(path), bgra)

    def _draw_overlay(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        masks: List[np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        overlay = image.copy()
        if overlay.dtype != np.uint8:
            overlay = (overlay * 255).astype(np.uint8)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        color = np.array((220, 20, 60), dtype=np.float32)
        for idx, box in enumerate(boxes):
            x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color.astype(np.uint8).tolist(), 2)
            cv2.putText(
                overlay,
                f"instance#{idx}",
                (x0, max(y0 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color.astype(np.uint8).tolist(),
                2,
                cv2.LINE_AA,
            )
            mask = masks[idx]
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask.astype(bool)
            blended = overlay[mask_bool].astype(np.float32) * (1 - self.config.overlay_alpha) + color * self.config.overlay_alpha
            overlay[mask_bool] = blended.astype(np.uint8)
        return overlay
