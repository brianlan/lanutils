import pytest
from numpy.testing import assert_almost_equal
import torch
import numpy as np

from ..iou import iou


def test_iou_empty_input():
    empty_out = iou(torch.tensor([]), torch.tensor([]))
    assert list(empty_out.shape) == [0, 1]
    assert empty_out.dim() == 2
    assert_almost_equal(empty_out.numpy(), np.array([]).reshape(-1, 1))


def test_iou_input_error():
    with pytest.raises(ValueError):
        _ = iou(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 7]]), torch.tensor([8, 8, 8, 8]))
    with pytest.raises(ValueError):
        _ = iou(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 7]]), torch.tensor([[[8, 8, 8, 8], [9,9,9,9]]]))


def test_iou():
    pass
