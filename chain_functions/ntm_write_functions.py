import numpy as np
from chainer.utils import type_check
from chainer import function


class NtmWriteErase(function.Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].ndim == 2,
            in_types[1].ndim == 2,
            in_types[2].ndim == 2,
            in_types[1].shape[1] == in_types[0].shape[0],
            in_types[2].shape[1] == in_types[0].shape[1],
        )

    def forward_cpu(self, inputs):
        mat_prev, w, e, = inputs
        mat = mat_prev - mat_prev * (w.T * e)
        return mat,

    def forward_gpu(self, inputs):
        pass

    def backward(self, inputs, grad_outputs):
        mat_prev, w, e = inputs
        gm = grad_outputs[0]
        gw = - e.dot((mat_prev * gm).T)
        ge = - w.dot(mat_prev * gm)
        gmat_prev = (np.ones(mat_prev.shape).astype(np.float32) - w.T.dot(e)) * gm
        return gmat_prev, gw, ge


class NtmWriteAdd(function.Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].ndim == 2,
            in_types[1].ndim == 2,
            in_types[2].ndim == 2,
            in_types[1].shape[1] == in_types[0].shape[0],
            in_types[2].shape[1] == in_types[0].shape[1],
        )

    def forward(self, inputs):
        return self.forward_cpu(inputs)

    def forward_cpu(self, inputs):
        mat_ers, w, a = inputs
        mat = mat_ers + w.T * a
        return mat,

    def forward_gpu(self, inputs):
        pass

    def backward(self, inputs, grad_outputs):
        mat_ers, w, a = inputs
        gm = grad_outputs[0]
        gw = a.dot(gm.T)
        ga = w.dot(gm)
        return gm, gw, ga


def ntm_write_erase(mat_prev, w, e):
    """
    Parameters:
        mat_prev : Previous memory matrix. N x M array.
        w : Head weighting. 1 x N array. Between 0,1 (open interval!) and sums to unit.
        e : Erase vector. 1 x M array. Between 0,1
    """
    return NtmWriteErase()(mat_prev, w, e)


def ntm_write_add(mat_ers, w, a):
    """
    Parameters:
        mat_ers : Memory matrix. N x M array.
        w : Head weighting. 1 x N array. Between 0,1 and sums to unit.
        a : Add vector. 1 x M array.
    """
    return NtmWriteAdd()(mat_ers, w, a)
