import pytest
import torch
import numpy as np

from ..modules import UpperBoundedReLU


def test_upper_bounded_relu_normal():
    input_data = torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32)
    relu = UpperBoundedReLU()
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))


def test_upper_bounded_relu_max_val():
    input_data = torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32)
    relu = UpperBoundedReLU(max_val=1.3)
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    relu.activate_upperbound()
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.3]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.3]]))
    relu.deactivate_upperbound()
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))


def test_upper_bounded_relu_track_running_mean_std():
    momentum = 0.1
    input_data = [torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32),
                  torch.tensor([[0, 0.2, -0.5, 0.5]], dtype=torch.float32),
                  torch.tensor([-2.1, -0.2, 0.0, -1.1], dtype=torch.float32)]

    relu = UpperBoundedReLU(momentum=momentum).train()

    # Round 1
    out = relu(input_data[0])

    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.running_mean.item(), 0.027500002)
    np.testing.assert_almost_equal(relu.running_std.item(), 0.127115443)

    # Round 2
    out = relu(input_data[1])

    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 0.2, 0, 0.5]]))
    np.testing.assert_almost_equal(relu.running_mean.item(), 0.029750001)
    np.testing.assert_almost_equal(relu.running_std.item(), 0.156435624)

    # Round 3
    out = relu(input_data[2])
    np.testing.assert_almost_equal(out.numpy(), np.array([0., 0., 0., 0.]))
    np.testing.assert_almost_equal(relu.running_mean.item(), -0.058224998)
    np.testing.assert_almost_equal(relu.running_std.item(), 0.236882284)

    another_data = torch.tensor([[1.1, 1.09], [-0.5, 0.2]], dtype=torch.float32)
    relu.eval().activate_upperbound()
    np.testing.assert_almost_equal(relu(another_data).numpy(), np.array([[0.295107282, 0.295107282], [0, 0.2]]))

    relu.train()
    out = relu(another_data)
    np.testing.assert_almost_equal(relu.running_mean.item(), -0.005152494)
    np.testing.assert_almost_equal(relu.running_std.item(), 0.290547669)
    np.testing.assert_almost_equal(out.numpy(), np.array([[0.295700163, 0.295700163], [0, 0.2]]))

    relu.eval()
    relu.max_val = 1.0
    np.testing.assert_almost_equal(relu(another_data).numpy(), np.array([[1.0, 1.0], [0, 0.2]]))

    relu.deactivate_upperbound()
    np.testing.assert_almost_equal(relu(another_data).numpy(), np.array([[1.1, 1.09], [0, 0.2]]))


def test_upper_bounded_relu_do_not_track():
    input_data = [torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32),
                  torch.tensor([[0, 0.2, -0.5, 0.5]], dtype=torch.float32),
                  torch.tensor([-2.1, -0.2, 0.0, -1.1], dtype=torch.float32)]

    relu = UpperBoundedReLU(track_running_stats=False).train()

    # Round 1
    out = relu(input_data[0])
    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 1.2], [0., 1.5]]))
    assert relu.running_mean is None
    assert relu.running_std is None

    # Round 2
    relu.eval().activate_upperbound()
    out = relu(input_data[1])
    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 0.2, 0, 0.5]]))
    assert relu.running_mean is None
    assert relu.running_std is None

    # Round 3
    relu.max_val = 0.15
    out = relu(input_data[2])
    np.testing.assert_almost_equal(out.numpy(), np.array([0, 0., 0, 0.0]))
    assert relu.running_mean is None
    assert relu.running_std is None
