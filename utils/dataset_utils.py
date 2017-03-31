import gzip

import h5py
import numpy as np

NOT_DATASET_MESSAGE = "dataset has to be tuple of 3 numpy.array or h5py.Dataset objects."
_CTR_LEARN_ON = 0
_CTR_SEQUENCE_END = 1

def is_dataset_type(dataset):
    return all([type(dataset) is tuple,
                len(dataset) is 3,
                all(map(lambda dlet: type(dlet) is np.ndarray or type(dlet) is h5py.Dataset, dataset)),
                dataset[0].shape[0] == dataset[1].shape[0] == dataset[2].shape[0]
                ])


def gen_sequence(seq_max_size, element_size):
    seq_size = np.random.randint(seq_max_size) + 1
    input_seq = np.asarray([np.random.randint(2, size=element_size) for i in range(seq_size)]) * 2 - 1
    input_seq[:, -1] = 0

    input_zeros = np.zeros((seq_size + 1, element_size))
    input_zeros[:, -1] = 0
    input_zeros[0, -1] = 1

    x = np.concatenate((input_seq, input_zeros)).astype(dtype=np.float32)
    y = np.concatenate((input_zeros, input_seq)).astype(dtype=np.float32)
    ctr = np.zeros((2 * seq_size + 1, 2)).astype(dtype=np.float32)
    ctr[seq_size + 1:, _CTR_LEARN_ON] = 1
    ctr[-1, _CTR_SEQUENCE_END] = 1
    return x, y, ctr


def gen_dataset(dataset_size, seq_max_size, element_size):
    x, y, ctr = gen_sequence(seq_max_size, element_size)
    for i in range(dataset_size - 1):
        nx, ny, nctr = gen_sequence(seq_max_size, element_size)
        x = np.concatenate((x, nx))
        y = np.concatenate((y, ny))
        ctr = np.concatenate((ctr, nctr))
    return x, y, ctr


def split_and_reshape_dataset(dataset, training_set_ratio):
    assert is_dataset_type(dataset), NOT_DATASET_MESSAGE
    x, y, ctr = dataset
    xh = x.reshape((x.shape[0], 1, x.shape[1])) * 0.7
    yh = y.reshape((y.shape[0], 1, y.shape[1])) * 0.7
    max_id = int(x.shape[0] * training_set_ratio)
    return xh[0:max_id, :, :], yh[0:max_id, :, :], ctr[0:max_id, :], \
           xh[max_id:, :, :], yh[max_id:, :, :], ctr[max_id:, :]


def save_dataset(path, filename, dataset):
    assert is_dataset_type(dataset), NOT_DATASET_MESSAGE
    x, y, ctr = dataset
    with gzip.open(path + "/" + filename + '.gzip', 'wb') as fil:
        np.save(fil, x)
        np.save(fil, y)
        np.save(fil, ctr)


def load_dataset(path, filename):
    with gzip.open(path + "/" + filename + '.gzip', 'rb') as fil:
        x = np.load(fil).astype(dtype=np.float32)
        y = np.load(fil).astype(dtype=np.float32)
        ctr = np.load(fil).astype(dtype=np.float32)
    return x, y, ctr


def get_sequence_ends(dataset_or_ctr):
    if is_dataset_type(dataset_or_ctr):
        ctr = dataset_or_ctr[2]
    elif type(dataset_or_ctr) is np.ndarray or type(dataset_or_ctr) is h5py.Dataset:
        ctr = dataset_or_ctr
    else:
        raise Exception("Input has to be either dataset type or ndarray or h5py.Dataset. It was "+str(type(dataset_or_ctr)))
    return np.where(ctr[:, _CTR_SEQUENCE_END] == 1)[0]


def is_sequence_end(ctr_vec):
    if ctr_vec.ndim == 1:
        return ctr_vec[_CTR_SEQUENCE_END] == 1
    raise Exception("control vector must be one dimensional, but it was "+str(ctr_vec.ndim)+ " dimensional.")


def is_learn_on(ctr_vec):
    if ctr_vec.ndim == 1:
        return ctr_vec[_CTR_LEARN_ON] == 1
    raise Exception("control vector must be one dimensional, but it was "+str(ctr_vec.ndim)+ " dimensional.")
