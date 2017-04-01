import unittest

from chainer import Variable
from chainer import gradient_check
from numpy.testing import *

from chain_functions.ntm_addressing_functions import *


class NtmAddressingTests(unittest.TestCase):
    @staticmethod
    def test_content_addressing():
        mat = Variable(np.asarray([[-2, -3, -1, -1], [2, 3, 1, 1], [-1, -1, 2, 3]], dtype=np.float32))
        key = Variable(np.asarray([[2, 3, 1, 1]], dtype=np.float32))
        beta = Variable(np.asarray([[3]], dtype=np.float32))
        w = ntm_content_addressing(mat, key, beta)
        assert_almost_equal(np.sum(w.data), 1, decimal=5, err_msg="weights doesn't sum to 1")
        assert_array_less(w.data[0, 0], w.data[0, 1], err_msg="key value is not selected")
        assert_array_less(w.data[0, 2], w.data[0, 1], err_msg="key value is not selected")

    @staticmethod
    def test_content_addressing_backward():
        bet = Variable(np.random.randn(1, 1).astype(np.float32))
        k = Variable(np.random.randn(1, 6).astype(np.float32))
        mat = Variable(np.random.randn(7, 6).astype(np.float32))
        y = ntm_content_addressing(mat, k, bet)
        y.grad = np.random.randn(1, 7).astype(np.float32)
        y.backward()

        f = lambda: (ntm_content_addressing(mat, k, bet).data,)
        gmat, gk, gbet = gradient_check.numerical_grad(f, (mat.data, k.data, bet.data), (y.grad,))
        gradient_check.assert_allclose(gbet, bet.grad, rtol=0.0001, atol=0.001)
        gradient_check.assert_allclose(gk, k.grad, rtol=0.0001, atol=0.001)
        gradient_check.assert_allclose(gmat, mat.grad, rtol=0.0001, atol=0.001)

    @staticmethod
    def test_select_interpolation():
        g = Variable(np.asarray([[0.4]], dtype=np.float32))
        w_key_similar = Variable(np.asarray([[1, 0, 0]], dtype=np.float32))
        w_previous = Variable(np.asarray([[0, 0, 1]], dtype=np.float32))
        w = ntm_select_interpolation(w_key_similar, w_previous, g)
        assert_almost_equal(np.sum(w.data), 1, decimal=5, err_msg="weights doesn't sum to 1")
        assert_array_almost_equal(np.asarray([[0.4, 0, 0.6]], dtype=np.float32), w.data)

    @staticmethod
    def test_select_interpolation_backward():
        g = Variable(np.random.randn(1, 1).astype(np.float32))
        w_sim = Variable(np.random.randn(1, 7).astype(np.float32))
        w_prev = Variable(np.random.randn(1, 7).astype(np.float32))
        y = ntm_select_interpolation(w_sim, w_prev, g)
        y.grad = np.random.randn(1, 7).astype(np.float32)
        y.backward()

        f = lambda: (ntm_select_interpolation(w_sim, w_prev, g).data,)
        gw_sim, gw_prev, gg = gradient_check.numerical_grad(f, (w_sim.data, w_prev.data, g.data), (y.grad,))
        gradient_check.assert_allclose(gw_sim, w_sim.grad, rtol=0.0001, atol=0.001)
        gradient_check.assert_allclose(gw_prev, w_prev.grad, rtol=0.0001, atol=0.001)
        gradient_check.assert_allclose(gg, g.grad, rtol=0.0001, atol=0.001)

    @staticmethod
    def test_convolutional_shift():
        s = Variable(np.asarray([[0, 0, 1]], dtype=np.float32))
        w_int = Variable(np.asarray([[1, 2, 3, 4]], dtype=np.float32))
        y = ntm_convolutional_shift(w_int, s)
        res = np.asarray([[2, 3, 4, 1]], dtype=np.float32)
        gradient_check.assert_allclose(res, y.data)

    @staticmethod
    def test_convolutional_shift_backward():
        s_data = np.random.randn(1, 7).astype(np.float32)
        s = Variable(s_data / np.sum(s_data))
        w_int = Variable(np.random.randn(1, 7).astype(np.float32))
        y = ntm_convolutional_shift(w_int, s)
        y.grad = np.random.randn(1, 7).astype(np.float32)
        y.backward()

        f = lambda: (ntm_convolutional_shift(w_int, s).data,)
        gw_int, gs = gradient_check.numerical_grad(f, (w_int.data, s.data), (y.grad,))
        gradient_check.assert_allclose(gw_int, w_int.grad, rtol=0.0001, atol=0.001)
        gradient_check.assert_allclose(gs, s.grad, rtol=0.0001, atol=0.001)

    @staticmethod
    def test_sharpening():
        gamma = Variable((np.random.randn(1, 1).astype(np.float32) + 1) * 2)
        w_data = np.abs(np.random.randn(1, 7).astype(np.float32))
        w_shifted = Variable(w_data / np.sum(w_data))
        y = ntm_sharpening(w_shifted, gamma)
        y.grad = np.random.randn(1, 7).astype(np.float32)
        y.backward()

        f = lambda: (ntm_sharpening(w_shifted, gamma).data,)
        gw_shifted, ggamma = gradient_check.numerical_grad(f, (w_shifted.data, gamma.data), (y.grad,))
        gradient_check.assert_allclose(ggamma, gamma.grad, rtol=0.0001, atol=0.001)
        gradient_check.assert_allclose(gw_shifted, w_shifted.grad, rtol=0.0001, atol=0.001)

    if __name__ == '__main__':
        unittest.main()
