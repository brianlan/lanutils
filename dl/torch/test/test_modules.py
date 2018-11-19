import pytest
import torch
import numpy as np

from ..modules import StatefulClippedReLU, StatefulHardtanh


def test_stateful_hardtanh_normal():
    input_data = torch.tensor([[-10.1, 1.2], [-0.5, 15.5]], dtype=torch.float32)
    htanh = StatefulHardtanh()
    np.testing.assert_almost_equal(htanh.train()(input_data).numpy(),
                                   np.array([[-10.1, 1.2], [-0.5, 15.5]]), decimal=6)
    np.testing.assert_almost_equal(htanh.eval()(input_data).numpy(),
                                   np.array([[-10.1, 1.2], [-0.5, 15.5]]), decimal=6)


def test_stateful_hardtanh_min_max_val():
    input_data = torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32)
    htanh = StatefulHardtanh(enforced_min=-0.8, enforced_max=1.3)
    np.testing.assert_almost_equal(htanh.train()(input_data).numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))
    np.testing.assert_almost_equal(htanh.eval()(input_data).numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))
    htanh.activate_boundary()
    np.testing.assert_almost_equal(htanh.train()(input_data).numpy(), np.array([[-0.8, 1.2], [-0.5, 1.3]]))
    np.testing.assert_almost_equal(htanh.eval()(input_data).numpy(), np.array([[-0.8, 1.2], [-0.5, 1.3]]))
    htanh.deactivate_boundary()
    np.testing.assert_almost_equal(htanh.train()(input_data).numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))
    np.testing.assert_almost_equal(htanh.eval()(input_data).numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))


def test_stateful_hardtanh_track_running_mean_std():
    momentum = 0.1
    input_data = [torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32),
                  torch.tensor([[0, 0.2, -0.5, 0.5]], dtype=torch.float32),
                  torch.tensor([-2.1, -0.2, 0.0, -1.1], dtype=torch.float32)]

    htanh = StatefulHardtanh(momentum=momentum).train()

    # Round 1
    out = htanh(input_data[0])
    np.testing.assert_almost_equal(out.numpy(), np.array([[-1.1, 1.2], [-0.5, 1.5]]))
    np.testing.assert_almost_equal(htanh.running_mean.item(), 0.027500002)
    np.testing.assert_almost_equal(htanh.running_std.item(), 0.127115443)

    # Round 2
    out = htanh(input_data[1])
    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 0.2, -0.5, 0.5]]))
    np.testing.assert_almost_equal(htanh.running_mean.item(), 0.029750001)
    np.testing.assert_almost_equal(htanh.running_std.item(), 0.156435624)

    # Round 3
    out = htanh(input_data[2])
    np.testing.assert_almost_equal(out.numpy(), np.array([-2.1, -0.2, 0.0, -1.1]))
    np.testing.assert_almost_equal(htanh.running_mean.item(), -0.058224998)
    np.testing.assert_almost_equal(htanh.running_std.item(), 0.236882284)

    another_data = torch.tensor([[1.1, 1.09], [-0.5, 0.2]], dtype=torch.float32)
    htanh.eval().activate_boundary()
    np.testing.assert_almost_equal(htanh(another_data).numpy(),
                                   np.array([[0.295107282, 0.295107282], [-0.295107282, 0.2]]))

    htanh.train()
    out = htanh(another_data)
    np.testing.assert_almost_equal(htanh.running_mean.item(), -0.005152494)
    np.testing.assert_almost_equal(htanh.running_std.item(), 0.290547669)
    np.testing.assert_almost_equal(out.numpy(), np.array([[0.295700163, 0.295700163], [-0.295700163, 0.2]]))

    htanh.eval()
    htanh.enforced_max = 1.0
    htanh.enforced_min = 0
    np.testing.assert_almost_equal(htanh(another_data).numpy(), np.array([[1.0, 1.0], [0, 0.2]]))

    htanh.deactivate_boundary()
    np.testing.assert_almost_equal(htanh(another_data).numpy(), np.array([[1.1, 1.09], [-0.5, 0.2]]))


def test_stateful_hardtanh_do_not_track():
    input_data = [torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32),
                  torch.tensor([[0, 0.2, -0.5, 0.5]], dtype=torch.float32),
                  torch.tensor([-2.1, 0.2, 0.5, -1.1], dtype=torch.float32)]

    htanh = StatefulHardtanh(track_running_stats=False).train()

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


def test_stateful_clipped_relu_normal():
    input_data = torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32)
    relu = StatefulClippedReLU()
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))


def test_stateful_clipped_relu_max_val():
    input_data = torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32)
    relu = StatefulClippedReLU(clip_at=1.3)
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    relu.activate_boundary()
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.3]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.3]]))
    relu.deactivate_boundary()
    np.testing.assert_almost_equal(relu.train()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))
    np.testing.assert_almost_equal(relu.eval()(input_data).numpy(), np.array([[0, 1.2], [0, 1.5]]))


def test_stateful_clipped_relu_track_running_mean_std():
    momentum = 0.1
    input_data = [torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32),
                  torch.tensor([[0, 0.2, -0.5, 0.5]], dtype=torch.float32),
                  torch.tensor([-2.1, -0.2, 0.0, -1.1], dtype=torch.float32)]

    relu = StatefulClippedReLU(momentum=momentum).train()

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
    relu.eval().activate_boundary()
    np.testing.assert_almost_equal(relu(another_data).numpy(), np.array([[0.295107282, 0.295107282], [0, 0.2]]))

    relu.train()
    out = relu(another_data)
    np.testing.assert_almost_equal(relu.running_mean.item(), -0.005152494)
    np.testing.assert_almost_equal(relu.running_std.item(), 0.290547669)
    np.testing.assert_almost_equal(out.numpy(), np.array([[0.295700163, 0.295700163], [0, 0.2]]))

    relu.eval()
    relu.enforced_max = 1.0
    np.testing.assert_almost_equal(relu(another_data).numpy(), np.array([[1.0, 1.0], [0, 0.2]]))

    relu.deactivate_boundary()
    np.testing.assert_almost_equal(relu(another_data).numpy(), np.array([[1.1, 1.09], [0, 0.2]]))


def test_stateful_clipped_relu_do_not_track():
    input_data = [torch.tensor([[-1.1, 1.2], [-0.5, 1.5]], dtype=torch.float32),
                  torch.tensor([[0, 0.2, -0.5, 0.5]], dtype=torch.float32),
                  torch.tensor([-2.1, -0.2, 0.0, -1.1], dtype=torch.float32)]

    relu = StatefulClippedReLU(track_running_stats=False).train()

    # Round 1
    out = relu(input_data[0])
    np.testing.assert_almost_equal(out.numpy(), np.array([[0, 1.2], [0., 1.5]]))
    assert relu.running_mean is None
    assert relu.running_std is None

    # Round 2
    relu.eval().activate_boundary()
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
