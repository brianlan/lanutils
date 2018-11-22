import collections

import torch
import torch.nn as nn
import torch.nn.functional as F


Boundary = collections.namedtuple('Boundary', 'lower upper')


class QClippedLayerWithInputStats(nn.Hardtanh):
    default_boundary = Boundary(-999999999., 999999999.)

    @staticmethod
    def _default_estimation_function():
        return lambda mu, std: (-(abs(mu) + 3 * abs(std)), abs(mu) + 3 * abs(std))

    def _register_tracking_state_attrs(self):
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.tensor(0.))
            self.register_buffer('running_std', torch.tensor(1.))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_std', None)
            self.register_parameter('num_batches_tracked', None)

    def __init__(self, momentum=0.1, enforced_min=None, enforced_max=None, track_running_stats=True,
                 estimation_function=None):
        super(QClippedLayerWithInputStats, self).__init__()
        self.enforced_min = enforced_min
        self.enforced_max = enforced_max
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.boundary_activated = False
        self._estimated_boundary = None
        self.estimation_function = estimation_function or self._default_estimation_function()
        self._register_tracking_state_attrs()

    def activate_boundary(self):
        self.boundary_activated = True
        return self

    def deactivate_boundary(self):
        self.boundary_activated = False
        return self

    @property
    def estimated_boundary(self):
        if self.training or self._estimated_boundary is None:
            if self.running_mean is not None and self.running_std is not None:
                self._estimated_boundary = Boundary(*self.estimation_function(self.running_mean, self.running_std))
        return self._estimated_boundary

    def _get_boundary_by_priority(self):
        """
        The priority is (from high to low):
            1. (enforced_min, enforced_max)
            2. (estimated_boundary.lower, estimated_boundary.upper)
            3. default_boundary
        """
        boundary = self.default_boundary
        if self.estimated_boundary is not None:
            boundary = Boundary(self.estimated_boundary.lower, self.estimated_boundary.upper)
        boundary = Boundary(boundary.lower if self.enforced_min is None else self.enforced_min,
                            boundary.upper if self.enforced_max is None else self.enforced_max)
        return boundary

    @staticmethod
    def _hardtanh(input, boundary):
        return F.hardtanh(input, boundary.lower, boundary.upper, False)

    def forward(self, input):
        if self.training and self.track_running_stats:
            self.running_mean += self.momentum * (input.mean().detach() - self.running_mean)
            self.running_std += self.momentum * (input.std().detach() - self.running_std)
            self.num_batches_tracked += torch.tensor(1, dtype = torch.long)
        if self.boundary_activated:
            return self._hardtanh(input, self._get_boundary_by_priority())
        return self._hardtanh(input, self.default_boundary)


class QClippedReLUWithInputStats(QClippedLayerWithInputStats):
    def __init__(self, momentum=0.1, clip_at=None, track_running_stats=True, estimation_function=None):
        super(QClippedReLUWithInputStats, self).__init__(momentum=momentum, enforced_min=0, enforced_max=clip_at,
                                                         track_running_stats=track_running_stats,
                                                         estimation_function=estimation_function)
        QClippedReLUWithInputStats.default_boundary = Boundary(0., self.default_boundary.upper)
