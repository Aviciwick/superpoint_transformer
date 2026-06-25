import os
import sys

import torch


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.utils.torchmetrics import SafeMeanMetric


def test_safe_mean_metric_returns_zero_before_first_update():
    metric = SafeMeanMetric()

    value = metric.compute()

    assert torch.equal(value, torch.tensor(0.0))


def test_safe_mean_metric_matches_mean_after_update():
    metric = SafeMeanMetric()
    metric.update(torch.tensor(2.0))
    metric.update(torch.tensor(4.0))

    value = metric.compute()

    assert torch.isclose(value, torch.tensor(3.0))
