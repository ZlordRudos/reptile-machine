import unittest

import chainer.functions
import numpy as np
from chainer import Variable
from numpy.testing import *

from links import ntm_one_head


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_splitting():
        ntm = ntm_one_head.NtmOneHead(3, 4, 1)
        e = np.asarray([1, 0, 0, 0])
        a = np.asarray([2, 2, 2, 2])
        key = np.asarray([0, 0, 0, 1])
        shift = np.asarray([0, 1, 0])
        beta = np.asarray([1])
        g = np.asarray([1])
        gamma = np.asarray([1])
        controls_tuple = (key, shift, beta, g, gamma, e, a)
        control_dat = np.concatenate((key, shift, beta, g, gamma, e, a)).astype(dtype=np.float32).reshape((1, 18))
        control = Variable(control_dat)
        spl = chainer.functions.array.split_axis.split_axis(control, np.asarray(ntm.sections), 1)
        i = 0
        for ctr in controls_tuple:
            assert_array_equal(ctr, spl[i].data[0])
            i += 1

    @staticmethod
    def test_control():
        ntm = ntm_one_head.NtmOneHead(3, 4, 1)  # w is set on [1,0,0]... the first vector of memory
        ntm.reset(init_mat=np.asarray([[1.0, 1.0, 1.0, 1.0],
                                       [-1.0, -1.0, -1.0, -1.0],
                                       [-1.0, 0.1, 1.0, -1.0]], dtype=np.float32))
        e = np.asarray([20, -20, -20, -20])  # erase first element of the selected vector
        a = np.asarray([10, 10, 10, 10])  # add positive value to each its element
        key = np.asarray([-5, 0, 9, -12])  # select third row
        shift = np.asarray(
            [10, -10, -10])  # mark third element of sel.vector as first element of shifted vector [2, 3, 3, 3]
        beta = np.asarray([20])  # enhancing similarity score
        g = np.asarray([10])  # only key selection matters
        gamma = np.asarray([1])
        controls_tuple = (key, shift, beta, g, gamma, e, a)
        control_dat = np.concatenate(controls_tuple).astype(dtype=np.float32).reshape((1, 18))
        control = Variable(control_dat)
        membit = ntm(control)
        assert_array_almost_equal(np.asarray([[0, 1, 1, 1]]) + np.tanh(10), membit.data, decimal=1)

    @staticmethod
    def test_backward_smoke():
        ntm = ntm_one_head.NtmOneHead(3, 4, 1)  # w is set on [1,0,0]... the first vector of memory [1, 1, 1, 1]
        ntm.reset(init_mat=np.asarray([[1.0, 1.0, 1.0, 1.0],
                                       [0.1, 0.1, 0.1, 0.1],
                                       [-1.0, 0.1, 1.0, -1.0]], dtype=np.float32))
        e = np.asarray([0.99, 0, 0, 0])  # erase first element of the selected vector [0, 1, 1, 1]
        a = np.asarray([2, 2, 2, 2])  # add 2 to each its element [2, 3, 3, 3]
        key = np.asarray([-5, 0, 9, -12])  # select third row [-1, 0, 1, -1]
        shift = np.asarray(
            [1, 0, 0])  # mark third element of sel.vector as first element of shifted vector [2, 3, 3, 3]
        beta = np.asarray([5])  # enhancing similarity score
        g = np.asarray([1])  # only key selection matters
        gamma = np.asarray([1])
        controls_tuple = (key, shift, beta, g, gamma, e, a)
        control_dat = np.concatenate(controls_tuple).astype(dtype=np.float32).reshape((1, 18))
        control = Variable(control_dat)
        membit = ntm(control)
        membit.grad = np.random.randn(1, 4).astype(np.float32)
        membit.backward()

    @staticmethod
    def test_zero_stability():
        ntm = ntm_one_head.NtmOneHead(3, 4, 1)
        control_dat = np.zeros((1, 18), dtype=np.float32)
        control_dat[0, 0] = 0.1  # key vector must not sum to zero, in practice it is very unlikely to happen (?)
        control = Variable(control_dat)
        membit = ntm(control)
        data = membit.data
        assert not np.isnan(np.sum(data))

    @staticmethod
    def test_100random_stability():
        ntm = ntm_one_head.NtmOneHead(3, 4, 1)
        control_dat = (np.random.random((1, 18)).astype(dtype=np.float32)-0.5)*100
        control = Variable(control_dat)
        membit = ntm(control)
        data = membit.data
        assert not np.isnan(np.sum(data))


if __name__ == '__main__':
    unittest.main()
