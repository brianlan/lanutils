import pytest
import torch
import numpy as np

from ..modules import QClippedLayerWithInputStats


@pytest.fixture
def estimation_function():
    return lambda mu, std: (-(abs(mu) + abs(std)), abs(mu) + abs(std))


def test_q_clipped_layer_normal(estimation_function):
    input_data = torch.tensor([[-10.1, 1.2], [-0.5, 15.5]], dtype=torch.float32)
    htanh = QClippedLayerWithInputStats(estimation_function=estimation_function)
    np.testing.assert_almost_equal(htanh.train()(input_data).numpy(),
                                   np.array([[-10.1, 1.2], [-0.5, 15.5]]), decimal=6)
    np.testing.assert_almost_equal(htanh.eval()(input_data).numpy(),
                                   np.array([[-10.1, 1.2], [-0.5, 15.5]]), decimal=6)


def test_q_clipped_layer_min_max_val(estimation_function):
    input_data = torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32)
    htanh = QClippedLayerWithInputStats(enforced_min=-0.8, enforced_max=1.3, estimation_function=estimation_function)
    np.testing.assert_almost_equal(htanh.train()(input_data).numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))
    np.testing.assert_almost_equal(htanh.eval()(input_data).numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))
    htanh.activate_boundary()
    np.testing.assert_almost_equal(htanh.train()(input_data).numpy(), np.array([[-0.8, 1.2], [-0.5, 1.3]]))
    np.testing.assert_almost_equal(htanh.eval()(input_data).numpy(), np.array([[-0.8, 1.2], [-0.5, 1.3]]))
    htanh.deactivate_boundary()
    np.testing.assert_almost_equal(htanh.train()(input_data).numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))
    np.testing.assert_almost_equal(htanh.eval()(input_data).numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))


def test_q_clipped_layer_track_running_mean_std(estimation_function):
    momentum = 0.1
    input_data = [torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32),
                  torch.tensor([[0, 0.2, -0.5, 0.5]], dtype=torch.float32),
                  torch.tensor([-2.1, -0.2, 0.0, -1.1], dtype=torch.float32)]

    htanh = QClippedLayerWithInputStats(momentum=momentum, estimation_function=estimation_function).train()

    # Round 1
    out = htanh(input_data[0])
    np.testing.assert_almost_equal(out.numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))
    np.testing.assert_almost_equal(htanh.running_mean.item(), 0.027500002)
    np.testing.assert_almost_equal(htanh.running_std.item(), 1.027115464)

    # Round 2
    out = htanh(input_data[1])
    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 0.2, -0.5, 0.5]]))
    np.testing.assert_almost_equal(htanh.running_mean.item(), 0.029750001)
    np.testing.assert_almost_equal(htanh.running_std.item(), 0.966435611)

    # Round 3
    out = htanh(input_data[2])
    np.testing.assert_almost_equal(out.numpy(), np.array([-2.1, -0.2, 0.0, -1.1]))
    np.testing.assert_almost_equal(htanh.running_mean.item(), -0.058224998)
    np.testing.assert_almost_equal(htanh.running_std.item(), 0.965882301)

    another_data = torch.tensor([[1.1, 1.09], [-0.5, 0.2]], dtype=torch.float32)
    htanh.eval().activate_boundary()
    np.testing.assert_almost_equal(htanh(another_data).numpy(),
                                   np.array([[1.024107299, 1.024107299], [-0.5, 0.2]]))

    htanh.train()
    out = htanh(another_data)
    np.testing.assert_almost_equal(htanh.running_mean.item(), -0.005152494)
    np.testing.assert_almost_equal(htanh.running_std.item(), 0.946647644)
    np.testing.assert_almost_equal(out.numpy(), np.array([[0.951800138, 0.951800138], [-0.5, 0.2]]))

    htanh.eval()
    htanh.enforced_max = 1.0
    htanh.enforced_min = 0
    np.testing.assert_almost_equal(htanh(another_data).numpy(), np.array([[1.0, 1.0], [0, 0.2]]))

    htanh.deactivate_boundary()
    np.testing.assert_almost_equal(htanh(another_data).numpy(), np.array([[1.1, 1.09], [-0.5, 0.2]]))


def test_q_clipped_layer_do_not_track(estimation_function):
    input_data = [torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32),
                  torch.tensor([[0, 0.2, -0.5, 0.5]], dtype=torch.float32),
                  torch.tensor([-2.1, 0.2, 0.5, -1.1], dtype=torch.float32)]

    htanh = QClippedLayerWithInputStats(track_running_stats=False, estimation_function=estimation_function).train()

    # Round 1
    out = htanh(input_data[0])
    np.testing.assert_almost_equal(out.numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))
    assert htanh.running_mean is None
    assert htanh.running_std is None

    # Round 2
    htanh.eval().activate_boundary()
    out = htanh(input_data[1])
    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 0.2, -0.5, 0.5]]))
    assert htanh.running_mean is None
    assert htanh.running_std is None

    # Round 3
    htanh.enforced_max = 0.3
    htanh.enforced_min = 0.15
    out = htanh(input_data[2])
    np.testing.assert_almost_equal(out.numpy(), np.array([0.15, 0.2, 0.3, 0.15]))
    assert htanh.running_mean is None
    assert htanh.running_std is None
