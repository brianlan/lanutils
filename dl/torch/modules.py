import torch
import torch.nn as nn
import torch.nn.functional as F


class UpperBoundedReLU(nn.Hardtanh):
    _FLOAT_MAX = 999999999.

    def __init__(self, momentum=0.1, max_val=None, track_running_stats=True, estimation_function=None):
        super(UpperBoundedReLU, self).__init__()
        self.max_val = max_val
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.upperbound_activated = False
        self._estimated_upperbound = None

        if estimation_function is None:
            self.estimation_function = lambda miu, std: abs(miu) + abs(std)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.tensor(0.))
            self.register_buffer('running_std', torch.tensor(0.))
            self.register_buffer('num_batches_tracked', torch.tensor(0.))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_std', None)
            self.register_parameter('num_batches_tracked', None)

    def activate_upperbound(self):
        self.upperbound_activated = True
        return self

    def deactivate_upperbound(self):
        self.upperbound_activated = False
        return self

    @property
    def estimated_upperbound(self):
        if self.training or self._estimated_upperbound is None:
            if self.running_mean is not None and self.running_std is not None:
                self._estimated_upperbound = self.estimation_function(self.running_mean, self.running_std)
        return self._estimated_upperbound

    def forward(self, input):
        if self.training and self.track_running_stats:
            self.running_mean = self.running_mean * (1 - self.momentum) + self.momentum * input.mean()
            self.running_std = self.running_std * (1 - self.momentum) + self.momentum * input.std()
        if self.upperbound_activated:
            upperbound = self.max_val or self.estimated_upperbound or self._FLOAT_MAX
            return F.hardtanh(input, 0, upperbound, False)
        return F.relu(input, False)
