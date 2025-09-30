"""Universal GroundingDINO + SAM 2 detector.

This module provides :class:`UniversalDetector`, a lightweight convenience
wrapper around GroundingDINO and SAM 2 that mirrors the tree-specific
pipeline shared by the user but removes any domain assumptions.  The class
manages checkpoint downloads, mask/overlay rendering and per-instance report
generation without pulling in additional dependencies such as CLIP.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch

from groundingdino.util import box_ops
from groundingdino.util.inference import load_image, load_model, predict
from huggingface_hub import hf_hub_download

from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


DEFAULT_PROMPT = "object . item . instance ."

# Grounding DINO constants
GROUNDING_REPO = "ShilongLiu/GroundingDINO"
GROUNDING_WEIGHTS = "groundingdino_swint_ogc.pth"
GROUNDING_CONFIG = "GroundingDINO_SwinT_OGC.cfg.py"

SAM2_REPO = "facebook/sam2-hiera-large"


@dataclass
class DetectionInstance:
    prompt: str
    phrase: str
    score: float
    bbox: np.ndarray
    mask: np.ndarray
    area_ratio: float


class UniversalDetector:
    """High-level wrapper around GroundingDINO and SAM 2.

    The detector mirrors the behaviour of the tree-specialised pipeline used by
    the user while removing any assumptions about the target class.  Each
    detection produces a dedicated folder inside the provided output directory
    with the overlay, mask, bounding-box crop and a JSON report describing the
    detection.
    """

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        models_dir: Path = Path("~/.autodetector"),
        overlay_color: tuple[int, int, int] = (34, 139, 34),
        mask_color: Optional[tuple[int, int, int]] = None,
        mask_alpha: int = 200,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self._models_dir = Path(models_dir).expanduser().resolve()
        self._models_dir.mkdir(parents=True, exist_ok=True)

        # Colours are provided as RGB for readability and converted to BGR when
        # drawing with OpenCV.
        if mask_color is None:
            mask_color = overlay_color
        self._overlay_color_bgr = np.array([overlay_color[2], overlay_color[1], overlay_color[0]], dtype=np.float32)
        self._mask_color_rgb = np.array(mask_color, dtype=np.uint8)
        self._mask_alpha = int(np.clip(mask_alpha, 0, 255))

        self._dino_model = self._load_groundingdino()
        self._sam_predictor = self._load_sam2()

    ################
    # LOAD METHODS #
    ################

    def _load_groundingdino(self):
        groundingdino_dir = self._models_dir / "groundingdino"
        groundingdino_dir.mkdir(parents=True, exist_ok=True)

        config_path = hf_hub_download(GROUNDING_REPO, GROUNDING_CONFIG, local_dir=groundingdino_dir)
        weights_path = hf_hub_download(GROUNDING_REPO, GROUNDING_WEIGHTS, local_dir=groundingdino_dir)

        model = load_model(str(config_path), str(weights_path))
        model.to(self.device)
        model.eval()
        return model

    def _load_sam2(self) -> SAM2ImagePredictor:
        sam2_dir = self._models_dir / "sam2"
        sam2_dir.mkdir(parents=True, exist_ok=True)

        config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[SAM2_REPO]
        checkpoint_path = hf_hub_download(SAM2_REPO, filename=checkpoint_name, local_dir=sam2_dir)

        sam_model = build_sam2(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=str(self.device),
        )
        return SAM2ImagePredictor(sam_model)

    #############
    # BASIC API #
    #############

    def detect(
        self,
        image_path: os.PathLike[str] | str,
        output_dir: os.PathLike[str] | str,
        top_k: int,
        *,
        prompt: str = DEFAULT_PROMPT,
        multimask_output: bool = False,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[Path]:
        """Run detection on ``image_path`` and materialise artifacts in ``output_dir``.

        Parameters
        ----------
        image_path:
            Path to the RGB image file to analyse.
        output_dir:
            Directory where per-instance folders will be created.
        top_k:
            Maximum number of detections to keep (after filtering by area).
        prompt:
            Text prompt forwarded to GroundingDINO.  Defaults to a generic prompt
            that favours prominent objects.
        multimask_output:
            If ``True`` the best mask predicted by SAM 2 will be selected among
            multiple proposals.
        """

        image_path = Path(image_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        instances = self.predict_instances(
            image_path=image_path,
            top_k=top_k,
            prompt=prompt,
            multimask_output=multimask_output,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if not instances:
            return []

        image_source, _ = load_image(str(image_path))
        height, width = image_source.shape[:2]

        created_instance_dirs: List[Path] = []

        for idx, instance in enumerate(instances):
            box_xyxy = instance.bbox
            mask_bool = instance.mask
            area_ratio = instance.area_ratio
            logit_score = instance.score
            phrase = instance.phrase
            x0_c = max(0, min(int(round(box_xyxy[0])), width - 1))
            y0_c = max(0, min(int(round(box_xyxy[1])), height - 1))
            x1_c = max(x0_c + 1, min(int(round(box_xyxy[2])), width))
            y1_c = max(y0_c + 1, min(int(round(box_xyxy[3])), height))

            x1_d = min(x1_c - 1, width - 1)
            y1_d = min(y1_c - 1, height - 1)

            instance_dir = output_dir / f"instance_{idx:02d}"
            instance_dir.mkdir(parents=True, exist_ok=True)

            overlay_bgr = self._draw_overlay(
                image_source,
                np.asarray([[x0_c, y0_c, x1_d, y1_d]], dtype=np.float32),
                [mask_bool.astype(np.uint8)],
            )
            overlay_path = instance_dir / "overlay.png"
            cv2.imwrite(str(overlay_path), overlay_bgr)

            mask_path = instance_dir / "mask.png"
            self._save_mask(mask_bool, mask_path)

            crop_rgb = image_source[y0_c:y1_c, x0_c:x1_c]
            if crop_rgb.size == 0:
                crop_rgb = image_source[max(0, y0_c) : max(0, y0_c + 1), max(0, x0_c) : max(0, x0_c + 1)]
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            bbox_path = instance_dir / "bbox.png"
            cv2.imwrite(str(bbox_path), crop_bgr)

            angle_deg = self._lean_angle(mask_bool)
            report = {
                "prompt": prompt,
                "phrase": (phrase or "").strip(),
                "score": float(logit_score),
                "bbox": [int(x0_c), int(y0_c), int(x1_c), int(y1_c)],
                "mask_area": int(mask_bool.sum()),
                "area_ratio": float(area_ratio),
                "lean_angle": None if angle_deg is None else float(angle_deg),
            }

            report_path = instance_dir / "report.json"
            with report_path.open("w", encoding="utf-8") as fp:
                json.dump(report, fp, ensure_ascii=False, indent=2)

            created_instance_dirs.append(instance_dir)

        return created_instance_dirs

    def predict_instances(
        self,
        *,
        image_path: os.PathLike[str] | str,
        top_k: int,
        prompt: str = DEFAULT_PROMPT,
        multimask_output: bool = False,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[DetectionInstance]:
        image_path = Path(image_path).expanduser().resolve()
        image_source, image_tensor = load_image(str(image_path))

        box_threshold = self.box_threshold if box_threshold is None else box_threshold
        text_threshold = self.text_threshold if text_threshold is None else text_threshold

        boxes, logits, phrases = predict(
            model=self._dino_model,
            image=image_tensor,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        if boxes.shape[0] == 0:
            return []

        height, width = image_source.shape[:2]
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes_xyxy *= torch.tensor([width, height, width, height], device=boxes_xyxy.device)
        boxes_xyxy = boxes_xyxy.cpu().numpy()
        logits_np = logits.sigmoid().cpu().numpy()

        self._sam_predictor.set_image(image_source)

        items: list[DetectionInstance] = []
        total_pixels = float(height * width) if height > 0 and width > 0 else 0.0
        for idx, (box_xyxy, logit_vec, phrase) in enumerate(zip(boxes_xyxy, logits_np, phrases)):
            masks_np, iou_predictions_np, _ = self._sam_predictor.predict(
                box=box_xyxy,
                multimask_output=multimask_output,
                normalize_coords=True,
            )
            if multimask_output and masks_np.shape[0] > 1:
                best_idx = int(np.argmax(iou_predictions_np))
                best_mask = masks_np[best_idx]
            else:
                best_mask = masks_np[0]

            mask_bool = best_mask.astype(bool)
            if total_pixels > 0:
                area_ratio = float(mask_bool.sum()) / total_pixels
            else:
                area_ratio = 0.0
            score = float(np.max(logit_vec))
            items.append(
                DetectionInstance(
                    prompt=prompt,
                    phrase=(phrase or "").strip(),
                    score=score,
                    bbox=box_xyxy,
                    mask=mask_bool,
                    area_ratio=area_ratio,
                )
            )

        items.sort(key=lambda instance: instance.area_ratio, reverse=True)
        if top_k > 0:
            items = items[:top_k]

        return items

    def process_directory(
        self,
        image_dir: os.PathLike[str] | str,
        output_dir: os.PathLike[str] | str,
        top_k: int,
        *,
        prompt: str = DEFAULT_PROMPT,
        multimask_output: bool = False,
        image_extensions: Optional[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp"),
        progress_callback: Callable[[int, int, Path], None] | None = None,
        result_callback: Callable[[int, int, Path, int], None] | None = None,
    ) -> List[List[Path]]:
        """Run detection over all images inside ``image_dir``.

        This mirrors :meth:`AutoLabeler.process_directory` to provide a lightweight
        loop that applies :meth:`detect` to each image in a folder while reporting
        progress through the optional callbacks.
        """

        image_dir_path = Path(image_dir).expanduser().resolve()
        output_dir_path = Path(output_dir).expanduser().resolve()
        output_dir_path.mkdir(parents=True, exist_ok=True)

        image_paths = [
            image_path
            for image_path in sorted(image_dir_path.iterdir())
            if image_path.is_file()
        ]
        if image_extensions is not None:
            allowed = tuple(ext.lower() for ext in image_extensions)
            image_paths = [
                image_path
                for image_path in image_paths
                if image_path.suffix.lower() in allowed
            ]

        total = len(image_paths)
        results: List[List[Path]] = []

        for index, image_path in enumerate(image_paths, start=1):
            if progress_callback is not None:
                progress_callback(index, total, image_path)

            instance_dirs = self.detect(
                image_path=image_path,
                output_dir=output_dir_path / image_path.stem,
                top_k=top_k,
                prompt=prompt,
                multimask_output=multimask_output,
            )
            results.append(instance_dirs)

            if result_callback is not None:
                result_callback(index, total, image_path, len(instance_dirs))

        return results

    ###########
    # HELPERS #
    ###########

    def _save_mask(self, mask: np.ndarray, path: Path) -> None:
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., 0] = self._mask_color_rgb[0]
        rgba[..., 1] = self._mask_color_rgb[1]
        rgba[..., 2] = self._mask_color_rgb[2]
        rgba[..., 3] = np.where(mask, self._mask_alpha, 0)
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

        color = self._overlay_color_bgr
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
            height, width = overlay.shape[:2]
            if mask_bool.shape != (height, width):
                # SAM occasionally returns masks with transposed dimensions when the
                # source image has been rotated through EXIF metadata.  Align the
                # mask with the image before applying the overlay.
                if mask_bool.T.shape == (height, width):
                    mask_bool = mask_bool.T
                else:
                    mask_resized = cv2.resize(
                        mask_bool.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask_bool = mask_resized.astype(bool)
            blended = overlay[mask_bool].astype(np.float32) * 0.5 + color * 0.5
            overlay[mask_bool] = blended.astype(np.uint8)
        return overlay

    @staticmethod
    def _lean_angle(mask: np.ndarray | None) -> float | None:
        if mask is None or not mask.any():
            return None

        coords = np.column_stack(np.nonzero(mask))
        if coords.shape[0] < 2:
            return None

        mean = coords.mean(axis=0)
        centered = coords - mean
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
        vertical = np.array([1.0, 0.0])
        normalized = principal_vector / np.linalg.norm(principal_vector)
        dot = abs(float(np.dot(normalized, vertical)))
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = float(np.arccos(dot))
        angle_deg = np.degrees(angle_rad)
        return angle_deg

