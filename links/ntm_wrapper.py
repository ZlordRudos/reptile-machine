import chainer
import chainer.functions as F
from chainer import Variable
from chainer import link

from chain_functions.ntm_write_functions import *
from ntm_heads import ReadHead, WriteHead, WriteReadHead


def get_read_head_control_vector_length(memory_cell_size, max_shift):
    return memory_cell_size + (2 * max_shift + 1) + 3


def get_write_head_control_vector_length(memory_cell_size, max_shift):
    return get_read_head_control_vector_length(memory_cell_size, max_shift) + 2 * memory_cell_size


def get_sections_head_controls(memory_cell_size, max_shift, head_order):
    write_len = get_write_head_control_vector_length(memory_cell_size, max_shift)
    read_len = get_read_head_control_vector_length(memory_cell_size, max_shift)
    sections = []
    prev = 0
    for head_type in head_order:
        if head_type is 'w' or head_type is 'wr':
            prev += write_len

        elif head_type is 'r':
            prev += read_len
        else:
            raise Exception("type >" + head_type + "< not supported")
        sections.append(prev)
    return sections


class NeuralTuringMachineWrapper(link.Chain):
    def __init__(self, controller, memory_size, memory_cell_size, max_shift,
                 head_order):
        super(NeuralTuringMachineWrapper, self).__init__(controller=controller)
        self.heads_order = head_order
        self.memory_size = memory_size
        self.memory_cell_size = memory_cell_size
        self.max_shift = max_shift
        self.init_mem = np.zeros((self.memory_size, self.memory_cell_size), dtype=np.float32) + 0.1
        self.init_h = None
        self.memory = None
        self.h = None
        self.heads = []
        self.number_of_read_heads = 0
        self.number_of_write_heads = 0
        self.number_of_writeread_heads = 0
        for head_type in head_order:
            if head_type is 'w':
                self.heads.append(WriteHead(memory_cell_size, max_shift))
                self.number_of_write_heads += 1
            elif head_type is 'r':
                self.heads.append(ReadHead(memory_cell_size, max_shift))
                self.number_of_read_heads += 1
            elif head_type is 'wr':
                self.heads.append(WriteReadHead(memory_cell_size, max_shift))
                self.number_of_writeread_heads += 1
            else:
                raise Exception("type >" + head_type + "< not supported")
        self.sections = get_sections_head_controls(memory_cell_size, max_shift, head_order)

    def get_init_memory(self):
        if self.init_mem.shape[0] != self.memory_size:
            self.init_mem = np.zeros((self.memory_size, self.memory_cell_size), dtype=np.float32) + 0.1
        return self.init_mem

    def get_init_h(self):
        if self.init_h is None:
            self.init_h = np.zeros((1, self.memory_cell_size *
                                    (self.number_of_writeread_heads + self.number_of_write_heads)),
                                   dtype=np.float32)
        return self.init_h

    def reset(self, new_memory_size=None):
        for head in self.heads:
            head.reset()
        if new_memory_size is not None:
            self.memory_size = new_memory_size
        self.memory = None
        self.h = None

    def __call__(self, x):
        if self.memory is None:
            self.memory = Variable(self.get_init_memory(), volatile='auto')
        if self.h is None:
            self.h = chainer.Variable(self.get_init_h(), volatile='auto')
        y = self.controller(F.concat([self.h, x]))
        y_s = F.split_axis(y, self.sections, 1)
        readouts = []
        for i in range(len(self.heads)):
            if self.heads_order[i] is 'w':
                self.memory = self.heads[i](self.memory, y_s[i])
            elif self.heads_order[i] is 'wr':
                readout, self.memory = self.heads[i](self.memory, y_s[i])
                readouts.append(readout)
            else:
                readout = self.heads[i](self.memory, y_s[i])
                readouts.append(readout)
        self.h = F.concat(readouts)
        return F.relu(y_s[-1])
