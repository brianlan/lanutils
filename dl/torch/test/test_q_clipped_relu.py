import pytest
import torch
import numpy as np

from ..modules import QClippedReLUWithInputStats


@pytest.fixture
def bound_estimate_func():
    return lambda mu, std: (-(abs(mu) + abs(std)), abs(mu) + abs(std))


def test_q_clipped_relu_normal(bound_estimate_func):
    input_data = torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32)
    relu = QClippedReLUWithInputStats(bound_estimate_func=bound_estimate_func)
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))


def test_q_clipped_relu_max_val(bound_estimate_func):
    input_data = torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32)
    relu = QClippedReLUWithInputStats(clip_at=1.3, bound_estimate_func=bound_estimate_func)
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    relu.activate_bound()
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.3]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.3]]))
    relu.deactivate_bound()
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))


def test_q_clipped_relu_track_running_mean_std(bound_estimate_func):
    momentum = 0.1
    input_data = [torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32),
                  torch.tensor([[0, 0.2, -0.5, 0.5]], dtype=torch.float32),
                  torch.tensor([-2.1, -0.2, 0.0, -1.1], dtype=torch.float32)]

    relu = QClippedReLUWithInputStats(momentum=momentum, bound_estimate_func=bound_estimate_func).train()

    # Round 1
    out = relu(input_data[0])

    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.running_mean.item(), 0.027500002)
    np.testing.assert_almost_equal(relu.running_std.item(), 1.027115464)

    # Round 2
    out = relu(input_data[1])

    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 0.2, 0, 0.5]]))
    np.testing.assert_almost_equal(relu.running_mean.item(), 0.029750001)
    np.testing.assert_almost_equal(relu.running_std.item(), 0.966435611)

    # Round 3
    out = relu(input_data[2])
    np.testing.assert_almost_equal(out.numpy(), np.array([0., 0., 0., 0.]))
    np.testing.assert_almost_equal(relu.running_mean.item(), -0.058224998)
    np.testing.assert_almost_equal(relu.running_std.item(), 0.965882301)

    another_data = torch.tensor([[1.1, 1.09], [-0.5, 0.2]], dtype=torch.float32)
    relu.eval().activate_bound()
    np.testing.assert_almost_equal(relu(another_data).numpy(), np.array([[1.024107299, 1.024107299], [0, 0.2]]))

    relu.train()
    out = relu(another_data)
    np.testing.assert_almost_equal(relu.running_mean.item(), -0.005152494)
    np.testing.assert_almost_equal(relu.running_std.item(), 0.946647644)
    np.testing.assert_almost_equal(out.numpy(), np.array([[0.951800138, 0.951800138], [0, 0.2]]))

    relu.eval()
    relu.enforced_max = 1.0
    np.testing.assert_almost_equal(relu(another_data).numpy(), np.array([[1.0, 1.0], [0, 0.2]]))

    relu.deactivate_bound()
    np.testing.assert_almost_equal(relu(another_data).numpy(), np.array([[1.1, 1.09], [0, 0.2]]))


def test_q_clipped_relu_do_not_track(bound_estimate_func):
    input_data = [torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32),
                  torch.tensor([[0, 0.2, -0.5, 0.5]], dtype=torch.float32),
                  torch.tensor([-2.1, -0.2, 0.0, -1.1], dtype=torch.float32)]

    relu = QClippedReLUWithInputStats(track_running_stats=False, bound_estimate_func=bound_estimate_func).train()

    # Round 1
    out = relu(input_data[0])
    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 1.2], [0., 1.5]]))
    assert relu.running_mean is None
    assert relu.running_std is None

    # Round 2
    relu.eval().activate_bound()
    out = relu(input_data[1])
    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 0.2, 0, 0.5]]))
    assert relu.running_mean is None
    assert relu.running_std is None

    # Round 3
    relu.enforced_max = 0.15
    out = relu(input_data[2])
    np.testing.assert_almost_equal(out.numpy(), np.array([0, 0., 0, 0.0]))
    assert relu.running_mean is None
    assert relu.running_std is None
