import chainer
import chainer.functions as F
from chainer import link
from chainer import report

from chain_functions.ntm_addressing_functions import *
from chain_functions.ntm_write_functions import *


def get_read_head_control_vector_length(memory_cell_size, max_shift):
    return memory_cell_size + (2 * max_shift + 1) + 3


def get_write_head_control_vector_length(memory_cell_size, max_shift):
    return get_read_head_control_vector_length(memory_cell_size, max_shift) + 2 * memory_cell_size


def get_sections_read_head(memory_cell_size, max_shift):
    shift_vec_size = 2 * max_shift + 1
    sections = [
        memory_cell_size,  # key
        memory_cell_size + shift_vec_size,  # shift
        memory_cell_size + shift_vec_size + 1,  # bet... key focus
        memory_cell_size + shift_vec_size + 2,  # g... gate for interpolation
        memory_cell_size + max_shift + 3  # gamma... sharpening coefficient
    ]
    return sections


def get_sections_write_head(memory_cell_size, max_shift):
    sections = get_sections_read_head(memory_cell_size, max_shift)
    sections.append(sections[-1] + memory_cell_size)  # e... erase vector
    sections.append(sections[-1] + memory_cell_size)  # a... add vector
    return sections


class BaseHead(link.Chain):
    def __init__(self, memory_cell_size, max_shift, **links):
        super(BaseHead, self).__init__(**links)
        self.memory_cell_size = memory_cell_size
        shift_vec_size = 2 * max_shift + 1
        self.shift_vec_size = shift_vec_size
        self.weighting = None

    @staticmethod
    def create_initial_weighting(memory_size):
        ret = np.zeros((1, memory_size), dtype=np.float32)
        ret[0, 0] = 1
        return ret

    def reset(self):
        self.weighting = None

    @staticmethod
    def normalize_movers(raw_key, raw_shift, raw_bet, raw_g, raw_gamma):
        bet = F.softplus(raw_bet)
        g = F.sigmoid(raw_g)
        shift = F.softmax(raw_shift)
        gamma = F.basic_math.add(F.softplus(raw_gamma), 1)
        key = F.tanh(raw_key)
        return key, shift, bet, g, gamma

    def move_head(self, memory, key, shift, bet, g, gamma):
        w_similar = ntm_content_addressing(memory, key, bet)
        w_interpolated = ntm_select_interpolation(w_similar, self.weighting, g)
        w_shifted = ntm_convolutional_shift(w_interpolated, shift)
        self.weighting = ntm_sharpening(w_shifted, gamma)
        report({'w': self.weighting.data, 'shift': shift.data, 'key': key.data, 'g': g.data},
               self)


class ReadHead(BaseHead):
    def __init__(self, memory_cell_size, max_shift, **links):
        super(ReadHead, self).__init__(memory_cell_size, max_shift, **links)
        self.sections = get_sections_read_head(memory_cell_size, max_shift)[:-1]

    def __call__(self, memory, controls):
        self.move_head(*self.normalize_movers(memory, *F.split_axis(controls, self.sections, 1)))
        return F.connection.linear.linear(self.weighting, chainer.functions.transpose(memory))


class WriteHead(BaseHead):
    def __init__(self, memory_cell_size, max_shift, **links):
        super(WriteHead, self).__init__(memory_cell_size, max_shift, **links)
        self.sections = get_sections_write_head(memory_cell_size, max_shift)[:-1]

    @staticmethod
    def normalize_writers(raw_e, raw_a):
        e = F.sigmoid(raw_e)
        a = F.tanh(raw_a)
        return e, a

    def write_into_memory(self, memory, e, a):
        mem_erased = ntm_write_erase(memory, self.weighting, e)
        return ntm_write_add(mem_erased, self.weighting, a)

    def __call__(self, memory, controls):
        raw_key, raw_shift, raw_bet, raw_g, raw_gamma, raw_e, raw_a = F.split_axis(controls, self.sections, 1)
        new_memory = self.write_into_memory(memory, *self.normalize_writers(raw_e, raw_a))
        self.move_head(new_memory, *self.normalize_movers(raw_key, raw_shift, raw_bet, raw_g, raw_gamma))
        return new_memory


class ReadWriteHead(WriteHead):
    def __call__(self, memory, controls):
        new_memory = super(ReadWriteHead, self).__call__(memory, controls)
        return F.connection.linear.linear(self.weighting, chainer.functions.transpose(new_memory)), new_memory
