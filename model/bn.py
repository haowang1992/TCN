from __future__ import division

import torch
from torch.nn import Module, init
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean_sketch', 'running_var_sketch', 'num_batches_tracked_sketch',
                     'running_mean_image', 'running_var_image', 'num_batches_tracked_image',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean_sketch', torch.zeros(num_features))
            self.register_buffer('running_var_sketch', torch.ones(num_features))
            self.register_buffer('num_batches_tracked_sketch', torch.tensor(0, dtype=torch.long))
            self.register_buffer('running_mean_image', torch.zeros(num_features))
            self.register_buffer('running_var_image', torch.ones(num_features))
            self.register_buffer('num_batches_tracked_image', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_sketch', None)
            self.register_parameter('running_var_sketch', None)
            self.register_parameter('num_batches_tracked_sketch', None)
            self.register_parameter('running_mean_image', None)
            self.register_parameter('running_var_image', None)
            self.register_parameter('num_batches_tracked_image', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean_sketch.zero_()
            self.running_var_sketch.fill_(1)
            self.num_batches_tracked_sketch.zero_()
            self.running_mean_image.zero_()
            self.running_var_image.fill_(1)
            self.num_batches_tracked_image.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked_sketch'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)
            num_batches_tracked_key = prefix + 'num_batches_tracked_image'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class _BatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.sketch_flag = True

    def forward(self, input):
        self._check_input_dim(input)

        if self.sketch_flag:
            # exponential_average_factor is set to self.momentum
            # (when it is available) only so that if gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked_sketch is not None:
                    self.num_batches_tracked_sketch = self.num_batches_tracked_sketch + 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked_sketch)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            return F.batch_norm(
                input, self.running_mean_sketch, self.running_var_sketch, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            # exponential_average_factor is set to self.momentum
            # (when it is available) only so that if gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked_image is not None:
                    self.num_batches_tracked_image = self.num_batches_tracked_image + 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked_image)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            return F.batch_norm(
                input, self.running_mean_image, self.running_var_image, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)


class MSSBN(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MSSBN, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class MSSBN2d(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MSSBN2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class MSSBN1d(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MSSBN1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))