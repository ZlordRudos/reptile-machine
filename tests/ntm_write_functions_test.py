import unittest
from chainer import Variable
from chainer import gradient_check
import numpy as np
from numpy.testing import *

from chain_functions.ntm_write_functions import ntm_write_erase, ntm_write_add


class NtmWriteTests(unittest.TestCase):
    @staticmethod
    def test_erase_middle():
        mat = Variable(np.asarray([[-2, -3, -1, -1], [2, 3, 1, 1], [-1, -1, 2, 3]], dtype=np.float32))
        w = Variable(np.asarray([[0, 1, 0]], dtype=np.float32))
        e = Variable(np.asarray([[0, 1, 1, 0]], dtype=np.float32))
        new_mat = ntm_write_erase(mat, w, e)
        result = np.asarray([[-2, -3, -1, -1], [2, 0, 0, 1], [-1, -1, 2, 3]], dtype=np.float32)
        assert_array_almost_equal(result, new_mat.data, decimal=5)

    @staticmethod
    def test_write_erase_backward():
        e = Variable(np.random.randn(1, 6).astype(np.float32))
        w = Variable(np.random.randn(1, 7).astype(np.float32))
        mat = Variable(np.random.randn(7, 6).astype(np.float32))
        y = ntm_write_erase(mat, w, e)
        y.grad = np.random.randn(7, 6).astype(np.float32)
        y.backward()

        f = lambda: (ntm_write_erase(mat, w, e).data,)
        gmat, gw, ge = gradient_check.numerical_grad(f, (mat.data, w.data, e.data), (y.grad,))
        gradient_check.assert_allclose(gw, w.grad, rtol=0.0001, atol=0.001)
        gradient_check.assert_allclose(ge, e.grad, rtol=0.0001, atol=0.001)
        gradient_check.assert_allclose(gmat, mat.grad, rtol=0.0001, atol=0.001)

    @staticmethod
    def test_add_middle():
        mat = Variable(np.asarray([[-2, -3, -1, -1], [2, 0, 0, 1], [-1, -1, 2, 3]], dtype=np.float32))
        w = Variable(np.asarray([[0, 1, 0]], dtype=np.float32))
        a = Variable(np.asarray([[1, 7, 7, 1]], dtype=np.float32))
        new_mat = ntm_write_add(mat, w, a)
        result = np.asarray([[-2, -3, -1, -1], [3, 7, 7, 2], [-1, -1, 2, 3]], dtype=np.float32)
        assert_array_almost_equal(result, new_mat.data, decimal=5)

    @staticmethod
    def test_write_add_backward():
        a = Variable(np.random.randn(1, 6).astype(np.float32))
        w = Variable(np.random.randn(1, 7).astype(np.float32))
        mat = Variable(np.random.randn(7, 6).astype(np.float32))
        y = ntm_write_add(mat, w, a)
        y.grad = np.random.randn(7, 6).astype(np.float32)
        y.backward()

        f = lambda: (ntm_write_add(mat, w, a).data,)
        gmat, gw, ga = gradient_check.numerical_grad(f, (mat.data, w.data, a.data), (y.grad,))
        gradient_check.assert_allclose(gw, w.grad, rtol=0.0001, atol=0.001)
        gradient_check.assert_allclose(ga, a.grad, rtol=0.0001, atol=0.001)
        gradient_check.assert_allclose(gmat, mat.grad, rtol=0.0001, atol=0.001)


if __name__ == '__main__':
    unittest.main(exit=False)
