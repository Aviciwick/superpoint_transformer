import torch
from torchmetrics import MeanMetric as TorchMeanMetric


class SafeMeanMetric(TorchMeanMetric):
    """MeanMetric that returns 0 before the first update.

    Lightning/torchmetrics may probe validation/test metrics before those
    metrics have seen any batches. The default MeanMetric compute path then
    yields NaN, which later gets re-aggregated into warning spam. Returning a
    neutral zero keeps logging stable without affecting updated metrics.
    """

    def compute(self) -> torch.Tensor:
        weight = getattr(self, "weight", None)
        if torch.is_tensor(weight) and weight.numel() > 0:
            total_weight = weight.detach()
            if total_weight.ndim > 0:
                total_weight = total_weight.sum()
            if float(total_weight.item()) == 0.0:
                return torch.zeros((), device=weight.device, dtype=torch.float32)
        return super().compute()
