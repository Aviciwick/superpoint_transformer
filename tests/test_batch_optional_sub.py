import torch

from src.data import Batch, Cluster, Data


def test_batch_from_data_list_drops_partial_optional_sub_cluster():
    with_sub = Data(
        pos=torch.randn(2, 3),
        sub=Cluster(
            torch.tensor([0, 2], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        ),
    )
    without_sub = Data(pos=torch.randn(3, 3))

    batch = Batch.from_data_list([with_sub, without_sub])

    assert batch.pos.shape[0] == 5
    assert batch.sub is None
