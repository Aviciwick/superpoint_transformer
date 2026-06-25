#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data import NAG  # noqa: E402
from src.manuscript_tools.semantic_boundary import (  # noqa: E402
    aggregate_semantic_boundary_stats,
    compute_semantic_boundary_scene_stats,
)


def _collect_prediction_files(pred_dir: str = None, pred_files=None):
    if pred_files:
        return [Path(path).resolve() for path in pred_files]
    if pred_dir is None:
        raise ValueError("Provide either --pred-dir or --pred-files")
    return sorted(Path(pred_dir).resolve().glob("*.h5"))


def _load_level0_prediction(path: Path):
    nag = NAG.load(str(path), low=0, high=0)
    level0 = nag[0]
    pred = getattr(level0, "semantic_pred", None)
    if pred is None and getattr(level0, "logits", None) is not None:
        pred = level0.logits.argmax(dim=1)
    if pred is None:
        raise ValueError(f"{path} does not contain level-0 semantic predictions")
    if getattr(level0, "y", None) is None:
        raise ValueError(f"{path} does not contain level-0 ground-truth labels")
    if getattr(level0, "pos", None) is None:
        raise ValueError(f"{path} does not contain level-0 coordinates")
    return level0.pos, level0.y, pred


def main():
    parser = argparse.ArgumentParser(
        description="Export traceable semantic boundary metrics from tracked prediction H5 files.",
    )
    parser.add_argument("--pred-dir", type=str, default=None, help="Directory containing tracked .h5 predictions.")
    parser.add_argument("--pred-files", nargs="+", default=None, help="Explicit tracked .h5 prediction files.")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--boundary-band-k", type=int, default=16)
    parser.add_argument(
        "--boundary-distance",
        type=float,
        default=None,
        help="Deprecated radius protocol. If set, use with --boundary-band-k -1 to force radius boundaries.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSON path or '-' for stdout.")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    files = _collect_prediction_files(pred_dir=args.pred_dir, pred_files=args.pred_files)
    if not files:
        raise SystemExit("No prediction files found.")

    per_scene = []
    scene_stats = []
    for path in files:
        coords, gt, pred = _load_level0_prediction(path)
        stats = compute_semantic_boundary_scene_stats(
            point_coords=coords,
            gt_semantic=gt,
            pred_semantic=pred,
            num_classes=args.num_classes,
            boundary_distance=args.boundary_distance,
            boundary_band_k=args.boundary_band_k if args.boundary_band_k > 0 else None,
        )
        scene_summary = aggregate_semantic_boundary_stats([stats])
        scene_stats.append(stats)
        per_scene.append(
            {
                "file": path.name,
                "overall_miou": scene_summary["overall_miou"],
                "transition_region_miou": scene_summary["transition_region_miou"],
                "boundary_iou": scene_summary["boundary_iou"],
                "boundary_f1": scene_summary["boundary_f1"],
                "boundary_precision": scene_summary["boundary_precision"],
                "boundary_recall": scene_summary["boundary_recall"],
            }
        )

    summary = aggregate_semantic_boundary_stats(scene_stats)
    payload = {
        "input": {
            "num_classes": args.num_classes,
            "boundary_band_k": args.boundary_band_k,
            "boundary_distance": args.boundary_distance,
            "files": [str(path) for path in files],
        },
        "summary": summary,
        "per_scene": per_scene,
    }

    text = json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False)
    if args.output == "-":
        print(text)
        return

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
