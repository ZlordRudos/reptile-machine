import numpy as np
from chainer.utils import type_check
from chainer import function


def _softmax(x):
    bc = np.max(x, axis=1)
    nju_x = (x.T - bc).T
    return (np.exp(nju_x).T / np.sum(np.exp(nju_x), axis=1)).T


def _softmax_backward(y, gy):
    gx = y * gy[0]
    sumdx = gx.sum(axis=1, keepdims=True)
    gx -= y * sumdx
    return gx


def _create_shift_matrix(vec, rows):
    return np.asarray([np.roll(vec, i) for i in range(rows)], dtype=np.float32).reshape(rows, vec.shape[1])


class NtmContentAddressing(function.Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].ndim == 2,
            in_types[1].ndim == 2,
            in_types[2].ndim == 2,
            in_types[1].shape[1] == in_types[0].shape[1],
            in_types[2].shape[1] == 1,
        )

    def forward_gpu(self, inputs):
        return self.forward_cpu(inputs)

    def forward_cpu(self, inputs):
        mat, k, bet = inputs
        self.norm_k = np.linalg.norm(k, ord=2, axis=1, keepdims=True)
        self.norm_mat = np.linalg.norm(mat, ord=2, axis=1, keepdims=True).T
        self.norm_mat_k = (self.norm_k * self.norm_mat)
        self.cos_sim = (k.dot(mat.T) / self.norm_mat_k)
        self.sfmx = _softmax(bet * self.cos_sim)
        return self.sfmx,

    def backward(self, inputs, grad_outputs):
        mat, k, bet = inputs
        gw = grad_outputs[0]
        sfmx_backward = _softmax_backward(self.sfmx, gw)
        gbet = sfmx_backward.dot(self.cos_sim.T)
        a = mat / self.norm_mat_k.T
        b = self.cos_sim.T.dot(k / (self.norm_k * self.norm_k))
        gk = sfmx_backward.dot(a - b) * bet
        gmat = sfmx_backward.T * (
            k / self.norm_mat_k.T - (self.cos_sim / (self.norm_mat * self.norm_mat)).T * mat) * bet
        return gmat, gk, gbet


class NtmSelectInterpolation(function.Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].ndim == 2,
            in_types[1].ndim == 2,
            in_types[2].ndim == 2,
            in_types[0].shape == in_types[1].shape,
            in_types[2].shape[1] == 1,
        )

    def forward_gpu(self, inputs):
        return self.forward_cpu(inputs)

    def forward_cpu(self, inputs):
        w_sim, w_prev, g = inputs
        return g * w_sim + (1 - g) * w_prev,

    def backward(self, inputs, grad_outputs):
        w_sim, w_prev, g = inputs
        gw = grad_outputs[0]
        gg = (w_sim - w_prev).dot(gw.T)
        gw_sim = g * gw
        gw_prev = (1 - g) * gw
        return gw_sim, gw_prev, gg


class NtmConvolutionalShift(function.Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[0].ndim == 2,
            in_types[1].ndim == 2,
            in_types[0].shape == in_types[1].shape,
        )

    def forward_gpu(self, inputs):
        return self.forward_cpu(inputs)

    def forward_cpu(self, inputs):
        w_int, s = inputs
        shift_mat = _create_shift_matrix(s, w_int.shape[1]).T
        return w_int.dot(shift_mat),

    def backward(self, inputs, grad_outputs):
        w_int, s = inputs
        gw_shift = _create_shift_matrix(grad_outputs[0], w_int.shape[1]).T
        gs = w_int.dot(gw_shift)
        gw_int = s.dot(gw_shift.T)
        return gw_int, gs


class NtmSharpening(function.Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[0].ndim == 2,
            in_types[1].ndim == 2,
            in_types[1].shape[1] == 1
        )

    def forward_gpu(self, inputs):
        return self.forward_cpu(inputs)

    def forward_cpu(self, inputs):
        w_shi, gmm = inputs
        self.w_gmm = np.power(w_shi, gmm.T)
        self.w_gmm_sum = np.sum(self.w_gmm, axis=1, keepdims=True)
        self.w = self.w_gmm / self.w_gmm_sum
        return self.w,

    def backward(self, inputs, grad_outputs):
        w_shi, gmm = inputs
        gw = grad_outputs[0]
        ln_w_shi = np.log(w_shi)
        ggmm = self.w * (ln_w_shi - (ln_w_shi.dot(self.w_gmm.T) / self.w_gmm_sum))
        ggmm = ggmm.dot(gw.T)
        w_der = (gmm * np.power(w_shi, (gmm - 1).T)) / self.w_gmm_sum
        gw_shi = (gw - (gw.dot(self.w.T))) * w_der
        return gw_shi, ggmm


def ntm_content_addressing(mat, key, beta):
    """
    Parameters:
        mat : Memory matrix. N x M array
        key : Key that is compared to each element of mat. 1 x M array
        beta : Key strength. 1 x 1 array. Positive number.
    """
    return NtmContentAddressing()(mat, key, beta)


def ntm_select_interpolation(w_key_similar, w_previous, g):
    """
    Parameters:
        w_key_similar : Similar content weighting. 1 x N array. Between 0,1 and sums to unit.
        w_previous : Previous weighting. 1 x N array. Between 0,1 and sums to unit.
        g : Interpolation gate. 1 x 1 array. Between 0 and 1.
    """
    return NtmSelectInterpolation()(w_key_similar, w_previous, g)


def ntm_convolutional_shift(w_interpolated, shift):
    """
    Parameters:
        w_interpolated : Interpolated weight. 1 x N array. Between 0,1 and sums to unit.
        shift : Shift weighting. 1 x N array. Between 0,1 and sums to unit.
    """
    return NtmConvolutionalShift()(w_interpolated, shift)


def ntm_sharpening(w_shifted, gamma):
    """
    Parameters:
        w_shifted : Shifted weight. 1 x N array. Between 0,1 and sums to unit.
        gamma : Sharpening coefficient. 1 x 1 array. Geq than 1.
    """
    return NtmSharpening()(w_shifted, gamma)
