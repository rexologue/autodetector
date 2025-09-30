from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from .detector import DetectionInstance, UniversalDetector


ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class RegionConfig:
    prompt: str
    top_k: int = 3
    box_threshold: float = 0.5
    text_threshold: float = 0.35
    multimask_output: bool = False


@dataclass
class ClassConfig:
    prompt: str
    region: str
    top_k: int = 5
    box_threshold: float = 0.5
    text_threshold: float = 0.35
    multimask_output: bool = False
    min_region_iou: float = 0.1


REGION_CONFIGS: Dict[str, RegionConfig] = {
    "trunk": RegionConfig(
        prompt="tree trunk on tree trunk . main stem on tree trunk . straight trunk on tree trunk .",
        top_k=3,
        box_threshold=0.48,
        text_threshold=0.33,
    ),
    "crown": RegionConfig(
        prompt="tree crown in tree crown . leafy canopy in tree crown . upper canopy in tree crown .",
        top_k=3,
        box_threshold=0.46,
        text_threshold=0.33,
    ),
}


CLASS_CONFIGS: Dict[str, ClassConfig] = {
    "crown_damage": ClassConfig(
        prompt=(
            "crown_damage . crown damage in tree crown . broken branches in tree crown . "
            "crown dieback in tree crown ."
        ),
        region="crown",
        min_region_iou=0.08,
    ),
    "fruiting_bodies": ClassConfig(
        prompt=(
            "fruiting_bodies . bracket fungi attached on tree trunk . polypore conks attached on tree trunk . "
            "fungal fruiting bodies attached on tree trunk ."
        ),
        region="trunk",
        min_region_iou=0.1,
    ),
    "hollows_on_trunk": ClassConfig(
        prompt=(
            "hollows_on_trunk . hollow opening on tree trunk . cavity entrance on tree trunk . "
            "natural cavity hole on tree trunk ."
        ),
        region="trunk",
        min_region_iou=0.1,
    ),
    "trunk_cracks": ClassConfig(
        prompt=(
            "trunk_cracks . vertical crack on tree trunk . longitudinal split on tree trunk . "
            "bark split on tree trunk ."
        ),
        region="trunk",
        box_threshold=0.55,
        min_region_iou=0.12,
    ),
    "trunk_damage": ClassConfig(
        prompt=(
            "trunk_damage . bark wound on tree trunk . bark missing area on tree trunk . "
            "mechanical damage on tree trunk ."
        ),
        region="trunk",
        min_region_iou=0.1,
    ),
    "trunk_rots": ClassConfig(
        prompt=(
            "trunk_rots . wood decay on tree trunk . heart rot on tree trunk . "
            "butt rot at base on tree trunk ."
        ),
        region="trunk",
        min_region_iou=0.1,
    ),
}


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum(dtype=np.float64)
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum(dtype=np.float64)
    if union == 0:
        return 0.0
    return float(intersection / union)


def union_masks(instances: Iterable[DetectionInstance]) -> Optional[np.ndarray]:
    masks: list[np.ndarray] = []
    for instance in instances:
        masks.append(instance.mask.copy())
    if not masks:
        return None
    combined = masks[0]
    for mask in masks[1:]:
        combined = np.logical_or(combined, mask)
    return combined


def detect_regions(detector: UniversalDetector, image_path: Path) -> Dict[str, dict]:
    regions: Dict[str, dict] = {}
    for region_name, config in REGION_CONFIGS.items():
        instances = detector.predict_instances(
            image_path=image_path,
            top_k=config.top_k,
            prompt=config.prompt,
            multimask_output=config.multimask_output,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
        )
        regions[region_name] = {
            "instances": instances,
            "mask": union_masks(instances),
        }
    return regions


def save_region_debug(
    detector: UniversalDetector,
    image_rgb: np.ndarray,
    output_dir: Path,
    region_name: str,
    instances: List[DetectionInstance],
) -> List[dict]:
    region_entries: List[dict] = []
    if not instances:
        return region_entries

    region_dir = output_dir / "regions" / region_name
    region_dir.mkdir(parents=True, exist_ok=True)

    height, width = image_rgb.shape[:2]
    for idx, instance in enumerate(instances):
        box_xyxy = instance.bbox
        x0_c = max(0, min(int(round(box_xyxy[0])), width - 1))
        y0_c = max(0, min(int(round(box_xyxy[1])), height - 1))
        x1_c = max(x0_c + 1, min(int(round(box_xyxy[2])), width))
        y1_c = max(y0_c + 1, min(int(round(box_xyxy[3])), height))

        x1_d = min(x1_c - 1, width - 1)
        y1_d = min(y1_c - 1, height - 1)

        instance_dir = region_dir / f"instance_{idx:02d}"
        instance_dir.mkdir(parents=True, exist_ok=True)

        overlay_bgr = detector._draw_overlay(  # type: ignore[attr-defined]
            image_rgb,
            np.asarray([[x0_c, y0_c, x1_d, y1_d]], dtype=np.float32),
            [instance.mask.astype(np.uint8)],
        )
        cv2.imwrite(str(instance_dir / "overlay.png"), overlay_bgr)

        detector._save_mask(instance.mask, instance_dir / "mask.png")  # type: ignore[attr-defined]

        crop_rgb = image_rgb[y0_c:y1_c, x0_c:x1_c]
        if crop_rgb.size == 0:
            crop_rgb = image_rgb[max(0, y0_c) : max(0, y0_c + 1), max(0, x0_c) : max(0, x0_c + 1)]
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(instance_dir / "bbox.png"), crop_bgr)

        report = {
            "prompt": instance.prompt,
            "phrase": instance.phrase,
            "score": instance.score,
            "bbox": [int(x0_c), int(y0_c), int(x1_c), int(y1_c)],
            "mask_area": int(instance.mask.sum()),
            "area_ratio": instance.area_ratio,
        }
        with (instance_dir / "report.json").open("w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)

        region_entries.append(report)

    return region_entries


def save_class_instances(
    detector: UniversalDetector,
    image_rgb: np.ndarray,
    output_dir: Path,
    class_name: str,
    instances: List[DetectionInstance],
) -> List[dict]:
    class_entries: List[dict] = []
    if not instances:
        return class_entries

    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    height, width = image_rgb.shape[:2]
    for idx, instance in enumerate(instances):
        box_xyxy = instance.bbox
        x0_c = max(0, min(int(round(box_xyxy[0])), width - 1))
        y0_c = max(0, min(int(round(box_xyxy[1])), height - 1))
        x1_c = max(x0_c + 1, min(int(round(box_xyxy[2])), width))
        y1_c = max(y0_c + 1, min(int(round(box_xyxy[3])), height))

        x1_d = min(x1_c - 1, width - 1)
        y1_d = min(y1_c - 1, height - 1)

        instance_dir = class_dir / f"instance_{idx:02d}"
        instance_dir.mkdir(parents=True, exist_ok=True)

        overlay_bgr = detector._draw_overlay(  # type: ignore[attr-defined]
            image_rgb,
            np.asarray([[x0_c, y0_c, x1_d, y1_d]], dtype=np.float32),
            [instance.mask.astype(np.uint8)],
        )
        cv2.imwrite(str(instance_dir / "overlay.png"), overlay_bgr)

        detector._save_mask(instance.mask, instance_dir / "mask.png")  # type: ignore[attr-defined]

        crop_rgb = image_rgb[y0_c:y1_c, x0_c:x1_c]
        if crop_rgb.size == 0:
            crop_rgb = image_rgb[max(0, y0_c) : max(0, y0_c + 1), max(0, x0_c) : max(0, x0_c + 1)]
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(instance_dir / "bbox.png"), crop_bgr)

        report = {
            "prompt": instance.prompt,
            "phrase": instance.phrase,
            "score": instance.score,
            "bbox": [int(x0_c), int(y0_c), int(x1_c), int(y1_c)],
            "mask_area": int(instance.mask.sum()),
            "area_ratio": instance.area_ratio,
        }
        with (instance_dir / "report.json").open("w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)

        class_entries.append(
            {
                "class": class_name,
                **report,
                "instance_dir": str(instance_dir),
            }
        )

    return class_entries


def process_image(
    detector: UniversalDetector,
    image_path: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    regions = detect_regions(detector, image_path)

    image_entry = {
        "image": str(image_path),
        "regions": {},
        "instances": [],
    }

    for region_name, payload in regions.items():
        region_instances = payload["instances"]
        region_reports = save_region_debug(detector, image_rgb, output_dir, region_name, region_instances)
        image_entry["regions"][region_name] = {
            "count": len(region_instances),
            "reports": region_reports,
        }

    for class_name, config in CLASS_CONFIGS.items():
        region_mask = regions.get(config.region, {}).get("mask") if regions else None
        instances = detector.predict_instances(
            image_path=image_path,
            top_k=config.top_k,
            prompt=config.prompt,
            multimask_output=config.multimask_output,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
        )
        if region_mask is not None:
            filtered_instances = []
            for instance in instances:
                iou = mask_iou(instance.mask, region_mask)
                if iou >= config.min_region_iou:
                    filtered_instances.append(instance)
        else:
            filtered_instances = instances

        class_reports = save_class_instances(detector, image_rgb, output_dir, class_name, filtered_instances)
        image_entry["instances"].extend(class_reports)

    return image_entry


def iterate_images(image_dir: Path) -> Iterable[Path]:
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        yield image_path


def build_dataset(detector: UniversalDetector, image_dir: Path, output_dir: Path) -> list[dict]:
    image_dir = image_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    image_paths = list(iterate_images(image_dir))
    total = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{total}] Processing {image_path.name}", flush=True)
        image_output_dir = output_dir / image_path.stem
        entry = process_image(detector, image_path, image_output_dir)
        manifest.append(entry)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
    print(f"Dataset written to {output_dir}. Summary: {summary_path}", flush=True)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tree defect dataset builder using GroundingDINO + SAM2")
    parser.add_argument("--images", type=Path, required=True, help="Directory with tree images")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write dataset artifacts")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory for caching model checkpoints (default ~/.autodetector)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    detector = UniversalDetector(
        device=args.device,
        box_threshold=0.5,
        text_threshold=0.35,
        models_dir=args.models_dir or Path("~/.autodetector"),
    )

    build_dataset(detector, args.images, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
