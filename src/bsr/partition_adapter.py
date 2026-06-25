from typing import Optional


__all__ = ["OptionalPartitionAdapter"]


class OptionalPartitionAdapter:
    """Optional wrapper around legacy dynamic partition refinement.

    This adapter is intentionally decoupled from the main BSR pipeline. It can
    be used in appendix-style exploration without becoming part of the core
    training or inference path.
    """

    def __init__(
        self,
        enable: bool = False,
        min_points_per_sp: int = 3,
        merge_small_clusters: bool = True,
    ):
        self.enable = bool(enable)
        self.min_points_per_sp = int(min_points_per_sp)
        self.merge_small_clusters = bool(merge_small_clusters)
        self._adapter: Optional[object] = None

        if self.enable:
            try:
                from src.hspt.spr import SuperpointRefiner

                self._adapter = SuperpointRefiner(
                    min_points_per_sp=self.min_points_per_sp,
                    merge_small_clusters=self.merge_small_clusters,
                )
            except Exception:
                self._adapter = None

    def __call__(self, *args, **kwargs):
        if not self.enable or self._adapter is None:
            return None
        return self._adapter(*args, **kwargs)
