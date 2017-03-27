import matplotlib.pyplot as plt
import numpy as np
import dataset_utils as DU
from os import listdir
from os.path import isfile, join
from chainer import Variable, Reporter, report, report_scope
import gzip
import h5py


def parse_file_name(filename):
    name_type = filename.split(".")
    ret = name_type[0].split("_")
    return ret, name_type[1]


def gen_file_name(spec_arr, file_type):
    return "_".join(spec_arr) + "." + file_type


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
    observation_agg_np = dict(map(lambda (k, v): (k, np.array(v)), observation_agg.iteritems()))
    observation_agg_np["inputs"] = x
    observation_agg_np["controls"] = ctr
    if t is type(np.ndarray):
        observation_agg_np["targets"] = t
    return observation_agg_np


def save_observations(path, filename, observations):
    f = h5py.File(join(path, filename), "w")
    for key in observations:
        f.create_dataset(key, data=observations[key])
    f.close()

