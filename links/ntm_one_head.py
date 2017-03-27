import chainer
import chainer.functions as F
from chainer import link
from chainer import variable
from chainer import Reporter, report, report_scope

from chain_functions.ntm_write_functions import *
from chain_functions.ntm_addressing_functions import *
from chainer.links.connection import linear

from model_interfaces import Resetable


def _get_control_vector_length(memory_size, memory_cell_size):
    return 3 * memory_cell_size + memory_size + 3


class NtmOneHead(link.Chain, Resetable):
    def __init__(self, memory_shape, **links):
        assert type(memory_shape) is tuple, "Memory shape is a tuple of length 2."
        assert len(memory_shape) == 2, "Memory shape is a tuple of length 2."
        super(NtmOneHead, self).__init__(**links)
        self.memory_shape = memory_shape
        self.sections = [
            self.memory_shape[1],  # erase
            self.memory_shape[1] * 2,  # add
            self.memory_shape[1] * 3,  # key
            self.memory_shape[1] * 3 + self.memory_shape[0],  # shift
            self.memory_shape[1] * 3 + self.memory_shape[0] + 1,  # bet... key focus
            self.memory_shape[1] * 3 + self.memory_shape[0] + 2  # g... gate for interpolation
            # self.memory_shape[1] * 3 + self.memory_shape[0] + 3  # gamma... sharpening coefficient

        ]
        self.reset()

    def create_empty_memory(self):
        # TODO: Musi se vyresit problem s nulama! Kvuli kosinove vzdalenosti.
        if self.init_mat is None:
            return np.zeros(self.memory_shape, dtype=np.float32) + 0.1
        return self.init_mat

    def create_initial_weighting(self):
        ret = np.zeros((1, self.memory_shape[0]), dtype=np.float32)
        ret[0, 0] = 1
        return ret

    def reset(self, init_mat=None):
        self.mat = self.weighting = None
        self.init_mat = init_mat

    def __call__(self, control):
        if self.mat is None:
            self.mat = variable.Variable(self.create_empty_memory(), volatile='auto')
        if self.weighting is None:
            self.weighting = variable.Variable(self.create_initial_weighting(), volatile='auto')
        raw_e, raw_a, raw_key, raw_shift, raw_bet, raw_g, raw_gamma = F.split_axis(control, self.sections, 1)
        # normalization
        e = F.sigmoid(raw_e)
        bet = F.softplus(raw_bet)
        g = F.sigmoid(raw_g)
        shift = F.softmax(raw_shift)
        gamma = F.basic_math.add(F.softplus(raw_gamma), 1)
        a = F.tanh(raw_a)
        key = F.tanh(raw_key)
        # body
        mat_erased = ntm_write_erase(self.mat, self.weighting, e)
        self.mat = ntm_write_add(mat_erased, self.weighting, a)
        w_similar = ntm_content_addressing(self.mat, key, bet)
        w_interpolated = ntm_select_interpolation(w_similar, self.weighting, g)
        w_shifted = ntm_convolutional_shift(w_interpolated, shift)
        self.weighting = ntm_sharpening(w_shifted, gamma)
        report({'e': e.data, 'a': a.data, 'shift': shift.data, 'key': key.data, 'g': g.data}, self)
        return F.connection.linear.linear(self.weighting, chainer.functions.transpose(self.mat))


class NtmOneHeadLayer(NtmOneHead):
    def __init__(self, in_size, memory_size, memory_cell_size, out_size):
        super(NtmOneHeadLayer, self). \
            __init__((memory_size, memory_cell_size),
                     upward=linear.Linear(in_size, _get_control_vector_length(memory_size,
                                                                              memory_cell_size) + out_size),
                     lateral=linear.Linear(memory_cell_size,
                                           _get_control_vector_length(memory_size,
                                                                      memory_cell_size) + out_size,
                                           nobias=True),
                     )
        self.control_vector_length = _get_control_vector_length(memory_size, memory_cell_size)
        self.h = None
        self.reset()

    def reset(self, init_mat=None):
        super(NtmOneHeadLayer, self).reset(init_mat)
        self.h = None

    def __call__(self, x):
        y = self.upward(x)
        if self.h is not None:
            y += self.lateral(self.h)
        ctr, output = F.split_axis(y, [self.control_vector_length], 1)
        self.h = super(NtmOneHeadLayer, self).__call__(ctr)
        return F.tanh(output)


class NtmOneHeadWrapper(NtmOneHead):
    def __init__(self, controller, memory_size, memory_cell_size):
        super(NtmOneHeadWrapper, self).__init__((memory_size, memory_cell_size), controller=controller)
        self.control_vector_length = _get_control_vector_length(memory_size, memory_cell_size)
        self.h = None
        self.reset()
        # self.output_activation = output_activation

    def reset(self, init_mat=None):
        super(NtmOneHeadWrapper, self).reset(init_mat)
        self.h = None

    def __call__(self, x):
        if self.h is None:
            self.h = chainer.Variable(np.zeros((1, self.memory_shape[1]), dtype=np.float32), volatile='auto')
        y = self.controller(F.concat([x, self.h]))
        ctr, output = F.split_axis(y, [self.control_vector_length], 1)
        self.h = super(NtmOneHeadWrapper, self).__call__(ctr)
        return F.relu(output)
        # return self.output_activation(output)