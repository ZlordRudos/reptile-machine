import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

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
