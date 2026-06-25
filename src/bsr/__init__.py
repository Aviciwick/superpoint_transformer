"""BSR-SPT: Boundary-aware Selective Refinement for Superpoint Transformer."""

from .diagnostics import aggregate_mixed_superpoint_records, build_mixed_superpoint_scene_records
from .partition_adapter import OptionalPartitionAdapter
from .pipeline import (
    BSRModule,
    BSROutput,
    build_candidate_point_cloud,
    build_packed_points,
    compute_bsr_losses,
    run_bsr,
)
from .refiner import PointSuperpointBoundaryRefiner, PointSuperpointMixtureRefiner, RefinerOutput
from .selector import BoundaryPriorSelector


__all__ = [
    "BoundaryPriorSelector",
    "PointSuperpointBoundaryRefiner",
    "PointSuperpointMixtureRefiner",
    "RefinerOutput",
    "OptionalPartitionAdapter",
    "BSROutput",
    "BSRModule",
    "build_candidate_point_cloud",
    "build_packed_points",
    "run_bsr",
    "compute_bsr_losses",
    "build_mixed_superpoint_scene_records",
    "aggregate_mixed_superpoint_records",
]
