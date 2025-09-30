from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import AutoLabeler, AutoLabelerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GroundingDINO + SAM2 auto-labeling pipeline")
    parser.add_argument("--images", type=Path, required=True, help="Directory with source images")
    parser.add_argument("--output", type=Path, required=True, help="Directory to store auto-label artifacts")
    parser.add_argument("--prompt", type=str, default=None, help="GroundingDINO text prompt")
    parser.add_argument("--box-threshold", type=float, default=None, help="GroundingDINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=None, help="GroundingDINO text threshold")
    parser.add_argument("--top-k", type=int, default=None, help="Keep at most K detections per image")
    parser.add_argument("--min-area-ratio", type=float, default=None, help="Filter masks smaller than this ratio")
    parser.add_argument("--multimask", action="store_true", help="Enable SAM2 multi-mask output")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Directory to cache checkpoints (default ~/.autodetector)"
    )
    parser.add_argument(
        "--support-manifest",
        type=Path,
        default=None,
        help="JSON manifest with support crops for few-shot CLIP filtering",
    )
    parser.add_argument("--clip-threshold", type=float, default=None, help="Reject predictions below this CLIP score")
    parser.add_argument("--support-limit", type=int, default=None, help="Limit number of support crops to load")
    parser.add_argument(
        "--clip-model",
        type=str,
        default=None,
        help="open_clip model name for few-shot filtering (default ViT-B-32)",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default=None,
        help="open_clip pretrained weights tag (default laion2b_s34b_b79k)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=None,
        help="Overlay alpha when drawing masks (0-1)",
    )
    return parser


def parse_config(args: argparse.Namespace) -> AutoLabelerConfig:
    config = AutoLabelerConfig()
    if args.prompt is not None:
        config.prompt = args.prompt
    if args.box_threshold is not None:
        config.box_threshold = args.box_threshold
    if args.text_threshold is not None:
        config.text_threshold = args.text_threshold
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.min_area_ratio is not None:
        config.min_area_ratio = args.min_area_ratio
    if args.multimask:
        config.multimask_output = True
    if args.device is not None:
        config.device = args.device
    if args.models_dir is not None:
        config.models_dir = args.models_dir
    if args.support_manifest is not None:
        config.support_manifest = args.support_manifest
    if args.clip_threshold is not None:
        config.clip_threshold = args.clip_threshold
    if args.support_limit is not None:
        config.support_limit = args.support_limit
    if args.clip_model is not None:
        config.clip_model = args.clip_model
    if args.clip_pretrained is not None:
        config.clip_pretrained = args.clip_pretrained
    if args.overlay_alpha is not None:
        config.overlay_alpha = args.overlay_alpha
    return config


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = parse_config(args)
    labeler = AutoLabeler(config)

    def progress_callback(index: int, total: int, image_path: Path) -> None:
        print(f"[{index}/{total}] Starting {image_path.name}", flush=True)

    def result_callback(index: int, total: int, image_path: Path, count: int) -> None:
        instance_word = "instance" if count == 1 else "instances"
        print(
            f"[{index}/{total}] Found {count} {instance_word} in {image_path.name}",
            flush=True,
        )

    results = labeler.process_directory(
        args.images,
        args.output,
        progress_callback=progress_callback,
        result_callback=result_callback,
    )

    summary = []
    image_paths = sorted(
        p for p in args.images.expanduser().resolve().iterdir() if p.is_file()
    )
    for image_path, instances in zip(image_paths, results):
        summary.append(
            {
                "image": str(image_path),
                "instances": [
                    {
                        "index": instance.index,
                        "score": instance.score,
                        "clip_score": instance.clip_score,
                        "bbox": instance.bbox,
                        "phrase": instance.phrase,
                        "area_ratio": instance.area_ratio,
                        "overlay": str(instance.overlay_path),
                        "mask": str(instance.mask_path),
                        "crop": str(instance.crop_path),
                    }
                    for instance in instances
                ],
            }
        )

    summary_path = args.output.expanduser().resolve() / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Auto-labeling complete. Summary written to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
