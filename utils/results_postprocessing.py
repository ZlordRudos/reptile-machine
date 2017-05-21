from os import listdir, makedirs
from os.path import isfile, join, isdir

import h5py
import matplotlib.pyplot as plt
import numpy as np
from chainer import Variable

import dataset_utils as DU

DATASET_PATH = join("resources", "datasets")
ROOT_MODEL_PATH = join("resources", "models")

def check_and_make_dirs(*dir_paths):
    for dir_path in dir_paths:
        if not isdir(dir_path):
            makedirs(dir_path)


def get_safe_filename(dir_path, file_name):
    if not isfile(join(dir_path, file_name)):
        return file_name
    for i in range(1000):
        new_file_name = "tmp" + str(i) + "_" + file_name
        if not isfile(join(dir_path, new_file_name)):
            return new_file_name
    return "tmp1000" + "_" + file_name


def safe_path(dir_path, no_overwrite=True, file_type=None, *args):
    check_and_make_dirs(dir_path)
    if file_type is None:
        file_name = gen_file_name(args[:-1], args[-1])
    else:
        file_name = gen_file_name(args, file_type)
    if no_overwrite:
        file_name = get_safe_filename(dir_path, file_name)
    return join(dir_path, file_name)


def parse_file_name(filename):
    name_type = filename.split(".")
    ret = name_type[0].split("_")
    return ret, name_type[1]


def gen_file_name(*spec_arr, **kwargs):
    if "file_type" in kwargs:
        return "_".join(spec_arr) + "." + kwargs["file_type"]
    return "_".join(spec_arr[:-1]) + "." + spec_arr[-1]


def open_csvs_into_one(dir_path, file_names):
    data_agg = []
    for file_name in file_names:
        f = open(join(dir_path, file_name), 'r')
        data_agg.append(np.loadtxt(f))
    return np.concatenate(tuple(data_agg))


def get_dir_filenames(dir_path):
    return [i for i in listdir(dir_path) if isfile(join(dir_path, i))]


def collect_observations(model, reporter, x, ctr, t=None):
    observation = {}
    observation_agg = {}
    outputs = []
    with reporter.scope(observation):
        for i in range(x.shape[0]):
            pr = model.predict(Variable(np.asarray(x[i, :]), volatile=True)).data
            for key in observation:
                if key not in observation_agg:
                    observation_agg[key] = []
                observation_agg[key].append(observation[key])
            outputs.append(pr[0, :])
            if DU.is_sequence_end(ctr[i, :]):
                model.reset()
    observation_agg["outputs"] = outputs
    observation_agg_np = dict(map(lambda (k, v): (k, np.asarray(v, dtype=np.float32)), observation_agg.iteritems()))
    observation_agg_np["inputs"] = x
    observation_agg_np["controls"] = ctr
    if t is type(np.ndarray):
        observation_agg_np["targets"] = t
    return observation_agg_np


def merge_observations(observations):
    return dict([(k, np.concatenate(tuple([observation[k] for observation in observations]))) for k in observations[0]])


def save_observations(path, filename, observations):
    f = h5py.File(join(path, filename), "w")
    for key in observations:
        f.create_dataset(key, data=observations[key])
    f.close()


def load_observations(path, filename):
    f = h5py.File(join(path, filename), "r")
    return f


def _n(input_array):
    if input_array.ndim > 2:
        return np.reshape(input_array, (input_array.shape[0], input_array.shape[-1]))
    return input_array


def show_inputs_outputs_weights(inputs, outputs, weights, erasers, adders, shifts, keys, gs, pdf_file=None):
    xy = np.concatenate((_n(inputs).T, _n(outputs).T))
    ea = np.concatenate((_n(erasers).T, _n(adders).T))
    shifts_keys_gs = np.concatenate((_n(shifts).T, _n(keys).T, _n(gs).T))
    f, axarr = plt.subplots(2, 2)
    axarr[1][0].matshow(ea)
    axarr[0][0].matshow(xy)
    axarr[0][1].matshow(_n(weights).T)
    axarr[1][1].matshow(shifts_keys_gs)
    if pdf_file is not None:
        pdf_file.savefig()
        plt.close()
    else:
        plt.show()


def generate_ntm_control_vector_overview(observations, start_sequence=0, number_of_sequences=None, pdf_file=None):
    seq_ends = DU.get_sequence_ends(observations["controls"])
    last_end = 0
    if number_of_sequences is None or number_of_sequences > seq_ends.shape[0]:
        last_sequence = seq_ends.shape[0]
    else:
        last_sequence = number_of_sequences

    for i in range(start_sequence, last_sequence):
        show_inputs_outputs_weights(
            inputs=observations["inputs"][last_end:seq_ends[i] + 1, :],
            outputs=observations["outputs"][last_end:seq_ends[i] + 1, :],
            weights=observations["ntm/w"][last_end:seq_ends[i] + 1, :],
            erasers=observations["ntm/e"][last_end:seq_ends[i] + 1, :],
            adders=observations["ntm/a"][last_end:seq_ends[i] + 1, :],
            shifts=observations["ntm/shift"][last_end:seq_ends[i] + 1, :],
            keys=observations["ntm/key"][last_end:seq_ends[i] + 1, :],
            gs=observations["ntm/g"][last_end:seq_ends[i] + 1, :],
            pdf_file=pdf_file
        )
        last_end = seq_ends[i] + 1
