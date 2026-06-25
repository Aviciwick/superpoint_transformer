#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.bsr import (  # noqa: E402
    aggregate_mixed_superpoint_records,
    build_mixed_superpoint_scene_records,
)
from src.data import NAG  # noqa: E402


def _collect_prediction_files(pred_dir: str = None, pred_files=None):
    if pred_files:
        return [Path(path).resolve() for path in pred_files]
    if pred_dir is None:
        raise ValueError("Provide either a directory or explicit files.")
    return sorted(Path(pred_dir).resolve().glob("*.h5"))


def _pair_prediction_files(baseline_dir, bsr_dir, baseline_files=None, bsr_files=None):
    baseline_paths = _collect_prediction_files(pred_dir=baseline_dir, pred_files=baseline_files)
    bsr_paths = _collect_prediction_files(pred_dir=bsr_dir, pred_files=bsr_files)
    baseline_map = {path.name: path for path in baseline_paths}
    bsr_map = {path.name: path for path in bsr_paths}
    common = sorted(set(baseline_map) & set(bsr_map))
    if not common:
        raise ValueError("No overlapping tracked prediction files were found between baseline and BSR runs.")
    return [(baseline_map[name], bsr_map[name]) for name in common]


def _load_pair(baseline_path: Path, bsr_path: Path):
    baseline_nag = NAG.load(str(baseline_path), low=0, high=1)
    bsr_nag = NAG.load(str(bsr_path), low=0, high=1)
    if baseline_nag[0].num_nodes != bsr_nag[0].num_nodes:
        raise ValueError(f"Point count mismatch for {baseline_path.name}")

    baseline_pred = getattr(baseline_nag[0], "semantic_pred", None)
    if baseline_pred is None and getattr(baseline_nag[0], "logits", None) is not None:
        baseline_pred = baseline_nag[0].logits.argmax(dim=1)
    bsr_pred = getattr(bsr_nag[0], "semantic_pred", None)
    if bsr_pred is None and getattr(bsr_nag[0], "logits", None) is not None:
        bsr_pred = bsr_nag[0].logits.argmax(dim=1)

    if baseline_pred is None or bsr_pred is None:
        raise ValueError(f"Missing semantic predictions in {baseline_path.name} or {bsr_path.name}")
    if getattr(bsr_nag[0], "y", None) is None or getattr(bsr_nag[0], "super_index", None) is None:
        raise ValueError(f"{bsr_path.name} is missing level-0 labels or super_index")
    if getattr(bsr_nag[0], "pos", None) is None:
        raise ValueError(f"{bsr_path.name} is missing level-0 positions")

    level1 = bsr_nag[1]
    return {
        "point_coords": bsr_nag[0].pos,
        "gt_labels": bsr_nag[0].y,
        "baseline_pred": baseline_pred,
        "bsr_pred": bsr_pred,
        "super_index": bsr_nag[0].super_index,
        "candidate_mask": getattr(level1, "bsr_candidate_mask", None),
        "assignment_entropy": getattr(level1, "bsr_assignment_entropy", None),
        "secondary_slot_mass": getattr(level1, "bsr_secondary_slot_mass", None),
        "slot_diversity": getattr(level1, "bsr_slot_diversity", None),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export mixed-superpoint diagnostics from paired baseline and BSR tracked prediction H5 files.",
    )
    parser.add_argument("--baseline-dir", type=str, default=None)
    parser.add_argument("--baseline-files", nargs="+", default=None)
    parser.add_argument("--bsr-dir", type=str, default=None)
    parser.add_argument("--bsr-files", nargs="+", default=None)
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

    paired_paths = _pair_prediction_files(
        baseline_dir=args.baseline_dir,
        bsr_dir=args.bsr_dir,
        baseline_files=args.baseline_files,
        bsr_files=args.bsr_files,
    )

    scene_records = []
    for baseline_path, bsr_path in paired_paths:
        loaded = _load_pair(baseline_path, bsr_path)
        scene_records.append(
            build_mixed_superpoint_scene_records(
                point_coords=loaded["point_coords"],
                gt_labels=loaded["gt_labels"],
                baseline_pred=loaded["baseline_pred"],
                bsr_pred=loaded["bsr_pred"],
                super_index=loaded["super_index"],
                num_classes=args.num_classes,
                boundary_distance=args.boundary_distance,
                boundary_band_k=args.boundary_band_k if args.boundary_band_k > 0 else None,
                candidate_mask=loaded["candidate_mask"],
                assignment_entropy=loaded["assignment_entropy"],
                secondary_slot_mass=loaded["secondary_slot_mass"],
                slot_diversity=loaded["slot_diversity"],
            )
        )

    payload = {
        "input": {
            "num_classes": args.num_classes,
            "boundary_band_k": args.boundary_band_k,
            "boundary_distance": args.boundary_distance,
            "pairs": [
                {"baseline": str(baseline_path), "bsr": str(bsr_path)}
                for baseline_path, bsr_path in paired_paths
            ],
        },
        "summary": aggregate_mixed_superpoint_records(scene_records),
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
