import collections

import torch
import torch.nn as nn
import torch.nn.functional as F


Bound = collections.namedtuple('Bound', 'lower upper')


class QClippedLayerWithInputStats(nn.Hardtanh):
    default_bound = Bound(-999999999., 999999999.)

    @staticmethod
    def _default_bound_estimate_func():
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
                 bound_estimate_func=None):
        super(QClippedLayerWithInputStats, self).__init__()
        self.enforced_min = enforced_min
        self.enforced_max = enforced_max
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.bound_activated = False
        self._estimated_bound = None
        self.bound_estimate_func = bound_estimate_func or self._default_bound_estimate_func()
        self._register_tracking_state_attrs()

    def activate_bound(self):
        self.bound_activated = True
        return self

    def deactivate_bound(self):
        self.bound_activated = False
        return self

    @property
    def estimated_bound(self):
        if self.training or self._estimated_bound is None:
            if self.running_mean is not None and self.running_std is not None:
                self._estimated_bound = Bound(*self.bound_estimate_func(self.running_mean, self.running_std))
        return self._estimated_bound

    def _get_bound_by_priority(self):
        """
        The priority is (from high to low):
            1. (enforced_min, enforced_max)
            2. (estimated_bound.lower, estimated_bound.upper)
            3. default_bound
        """
        bound = self.default_bound
        if self.estimated_bound is not None:
            bound = Bound(self.estimated_bound.lower, self.estimated_bound.upper)
        bound = Bound(bound.lower if self.enforced_min is None else self.enforced_min,
                      bound.upper if self.enforced_max is None else self.enforced_max)
        return bound

    @staticmethod
    def _hardtanh(input, bound):
        return F.hardtanh(input, bound.lower, bound.upper, False)

    def forward(self, input):
        if self.training and self.track_running_stats:
            self.running_mean += self.momentum * (input.mean().detach() - self.running_mean)
            self.running_std += self.momentum * (input.std().detach() - self.running_std)
            self.num_batches_tracked += torch.tensor(1, dtype = torch.long)
        if self.bound_activated:
            return self._hardtanh(input, self._get_bound_by_priority())
        return self._hardtanh(input, self.default_bound)


class QClippedReLUWithInputStats(QClippedLayerWithInputStats):
    def __init__(self, momentum=0.1, clip_at=None, track_running_stats=True, bound_estimate_func=None):
        super(QClippedReLUWithInputStats, self).__init__(momentum=momentum, enforced_min=0, enforced_max=clip_at,
                                                         track_running_stats=track_running_stats,
                                                         bound_estimate_func=bound_estimate_func)
        QClippedReLUWithInputStats.default_bound = Bound(0., self.default_bound.upper)
