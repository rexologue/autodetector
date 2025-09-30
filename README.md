# GroundingDINO + SAM2 defect auto-labeling

This repository packages a lightweight pipeline that combines **GroundingDINO** and **SAM2**
for automatic annotation bootstrapping / pseudo-label generation. The goal is to turn a
small set of reference defect examples (few-shot) into usable annotations for the rest of
an unlabeled dataset.

## Key features

* ⚙️ One-line interface via `AutoLabeler` class or CLI.
* 🗂️ Generates per-instance folders with mask/box overlays, crops and JSON metadata.
* 🧠 Optional CLIP-based few-shot filtering using a tiny support manifest of positive crops.
* 🧾 Emits `summary.json` + per-image manifests for downstream dataset conversion.
* 🌳 Need a barebones GroundingDINO + SAM 2 wrapper without CLIP? Use `UniversalDetector`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adapt for your CUDA
pip install opencv-python pillow groundingdino-py sam2 open-clip-torch huggingface_hub
```

> **Note**: install the CUDA-enabled PyTorch build that matches your driver. On CPU-only
> machines use the default PyPI wheels instead.

The pipeline downloads all checkpoints on first use into `~/.autodetector` (or the custom
cache directory you pass).

## Few-shot support manifest

To enable CLIP-based filtering, prepare a JSON list describing a handful of positive
examples. Each entry should contain a path to the original support image and the bounding
box of a defect crop in **absolute pixel coordinates**:

```json
[
  {
    "image": "support/image_001.jpg",
    "bbox": [120, 45, 260, 220],
    "label": "crack"
  },
  {
    "image": "support/image_017.jpg",
    "bbox": [80, 66, 150, 145],
    "label": "corrosion"
  }
]
```

Only a few entries (4-10) are typically needed. The CLIP filter keeps detections whose
cosine similarity with the support embeddings exceeds `clip_threshold` (default `0.22`).
You can tune this value based on validation results.

## Python usage

```python
from pathlib import Path
from autodetector import AutoLabeler, AutoLabelerConfig, UniversalDetector

config = AutoLabelerConfig(
    prompt="defect . damage . crack .",
    support_manifest=Path("support_manifest.json"),
    box_threshold=0.28,
    text_threshold=0.22,
    min_area_ratio=0.0004,
    top_k=15,
)
labeler = AutoLabeler(config)

instances_per_image = labeler.process_directory(
    image_dir=Path("data/unlabeled"),
    output_dir=Path("outputs/autolabel"),
)

# Or use UniversalDetector for a prompt-driven pipeline that mirrors the
# tree-focused example shared in the issue but works for any concept and keeps
# the dependency footprint minimal.
detector = UniversalDetector()
detector.detect(
    image_path="data/example.jpg",
    output_dir="outputs/detections/example",
    top_k=10,
    prompt="bolt . screw . component .",
)
```

After running you will obtain the following structure:

```
outputs/autolabel/
├── IMG_0001/
│   ├── instance_00/
│   │   ├── bbox.png
│   │   ├── mask.png
│   │   ├── overlay.png
│   │   └── report.json
│   ├── instance_01/
│   └── manifest.json
├── IMG_0002/
└── summary.json
```

Each `report.json` contains the model scores, CLIP similarity (if available) and paths to
other assets to ease downstream filtering. `summary.json` is a convenient collection of
per-image detections.

## Command line interface

```bash
python -m autodetector.cli \
  --images data/unlabeled \
  --output outputs/autolabel \
  --prompt "defect . crack . corrosion ." \
  --support-manifest support_manifest.json \
  --box-threshold 0.3 \
  --text-threshold 0.25 \
  --min-area-ratio 0.0004 \
  --top-k 20
```

Use `--multimask` to let SAM2 return multiple masks per detection (the pipeline will pick
the best IoU prediction). When running on CPU or low-memory GPUs consider lowering
`top_k` and increasing `box_threshold`.

## Next steps

* Review overlays and tweak the thresholds. Increasing `clip_threshold` raises precision.
* Convert manifests to YOLO/COCO (all metadata is already available in JSON).
* Mix pseudo-labels with your 420 hand-labeled samples and fine-tune a compact detector
  such as YOLOv12x or RT-DETR.

Happy auto-labeling!
