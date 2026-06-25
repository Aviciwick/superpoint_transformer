from typing import Iterable, Optional

import torch


__all__ = ["extract_selector_handcrafted_features", "mean_pool_level0_feature"]


def mean_pool_level0_feature(
    nag,
    key: str,
    num_superpoints: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Pool a level-0 point attribute to level-1 superpoints by mean."""
    if not hasattr(nag[0], "super_index") or nag[0].super_index is None:
        return None
    if not hasattr(nag[0], key) or getattr(nag[0], key) is None:
        return None

    values = getattr(nag[0], key)
    if values is None or values.numel() == 0:
        return None
    if values.dim() == 1:
        values = values.unsqueeze(-1)

    super_index = nag[0].super_index.to(device=device, dtype=torch.long)
    if super_index.numel() != values.shape[0]:
        return None

    values = values.to(device=device, dtype=torch.float32)
    pooled = torch.zeros(num_superpoints, values.shape[1], device=device, dtype=values.dtype)
    counts = torch.zeros(num_superpoints, 1, device=device, dtype=values.dtype)
    pooled.index_add_(0, super_index, values)
    counts.index_add_(
        0,
        super_index,
        torch.ones(values.shape[0], 1, device=device, dtype=values.dtype),
    )
    return pooled / counts.clamp(min=1.0)


def extract_selector_handcrafted_features(
    nag,
    preferred_keys: Iterable[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Build a fixed-order handcrafted feature tensor for BSR selector scoring.

    Priority per feature key:
    1. direct level-1 attribute
    2. level-1 log_{length,surface,volume} fallback for size-like terms
    3. mean-pooled level-0 attribute using level-0 super_index
    4. zero fill if still unavailable
    """
    sp_data = nag[1]
    num_superpoints = sp_data.num_nodes if hasattr(sp_data, "num_nodes") else sp_data.pos.shape[0]
    pooled_level0 = {}
    features = []
    missing_keys = []

    for key in preferred_keys:
        attr = None
        if hasattr(sp_data, key) and getattr(sp_data, key) is not None:
            attr = getattr(sp_data, key)
        elif key in {"length", "surface", "volume"}:
            log_key = f"log_{key}"
            if hasattr(sp_data, log_key) and getattr(sp_data, log_key) is not None:
                attr = getattr(sp_data, log_key)
        else:
            if key not in pooled_level0:
                pooled_level0[key] = mean_pool_level0_feature(
                    nag=nag,
                    key=key,
                    num_superpoints=num_superpoints,
                    device=device,
                )
            attr = pooled_level0[key]

        if attr is None:
            missing_keys.append(key)
            attr = torch.zeros(num_superpoints, 1, device=device)
        elif attr.dim() == 1:
            attr = attr.unsqueeze(-1)

        features.append(attr.to(device=device, dtype=torch.float32))

    return torch.cat(features, dim=-1), missing_keys
