import datetime

import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
import numpy as np
from chainer import Reporter
from chainer import optimizers
from chainer import serializers
from matplotlib.backends.backend_pdf import PdfPages

import dataset_utils as DU
import learning_utils as LU
import results_postprocessing as RP
from links.ntm_one_head import NtmOneHeadLayer
from links.ntm_one_head import _get_control_vector_length, NtmOneHeadWrapper
from model_interfaces import NeuralNetworkModel

JNR = "/"
DATASET_PATH = "resources/datasets"
MODEL_PATH = "resources/models/exp1"
RESULTS_PATH = MODEL_PATH + JNR + "results"
PARAMETER_VIS_PATH = RESULTS_PATH + JNR + "parameter_vis"
LOSSES_PATH = RESULTS_PATH + JNR + "losses"
OBSERVATIONS_PATH = RESULTS_PATH + JNR + "observations"

TIMESTAMP = datetime.datetime.now().strftime("%y%m%d%H%M")


class ReluForwardController(chainer.Chain):
    def __init__(self, input_len, hidden_size, memory_size):
        super(ReluForwardController, self).__init__(l0=F.Linear(input_len * 2, hidden_size),
                                                    l1=F.Linear(hidden_size,
                                                                _get_control_vector_length(memory_size,
                                                                                           input_len) + input_len))

    def __call__(self, x):
        h1 = F.tanh(self.l0(x))
        return F.tanh(self.l1(h1))


class MSEError(chainer.Chain, NeuralNetworkModel):
    def __init__(self, ntm_wrapper):
        assert isinstance(ntm_wrapper, NtmOneHeadWrapper)
        super(MSEError, self).__init__(ntm=ntm_wrapper)

    def reset(self):
        self.ntm.reset()

    def predict(self, inp):
        return self.ntm(inp)

    def get_ntm_weighting(self):
        return self.ntm.weighting.data

    def get_ntm_memory(self):
        return self.ntm.mat.data

    def __call__(self, inp, t):
        oup = self.predict(inp)
        return F.mean_squared_error(oup, t)


class Model(chainer.Chain, NeuralNetworkModel):
    def __init__(self, input_len, hidden_size1, hidden_size2, memory_size):
        super(Model, self).__init__(l0=F.Linear(input_len, hidden_size1),
                                    l1=F.Linear(hidden_size1, hidden_size2),
                                    l2=NtmOneHeadLayer(hidden_size2, memory_size, input_len, input_len))

    def reset(self):
        self.l2.reset()

    def predict(self, inp):
        h0 = F.relu(self.l0(inp))
        h1 = F.relu(self.l1(h0))
        oup = F.relu(self.l2(h1))
        return oup

    def __call__(self, inp, t):
        oup = self.predict(inp)
        return F.mean_squared_error(oup, t)


def create_model(element_size, hidden_size, memory_size):
    model = Model(element_size, hidden_size, hidden_size, memory_size)
    optimizer = optimizers.RMSpropGraves()
    optimizer.setup(model)
    optimizer.lr = 10e-5
    return model, optimizer


def create_model2(element_size, hidden_size, memory_size, learning_rate=10e-5):
    model = MSEError(
        NtmOneHeadWrapper(ReluForwardController(element_size, hidden_size, memory_size), memory_size, element_size))
    optimizer = optimizers.RMSpropGraves()
    optimizer.setup(model)
    optimizer.lr = learning_rate
    return model, optimizer


######################################################################
######################################################################
learning_rate = 10e-4
train_type = 0  # 0 - basic
load_model = True
debug_train = True
# train = False
train = True
# show_pics = True
show_pics = False
show_last_losses = False
collect_observations = False
generate_dataset = False
model_name = "test_relu3"
dataset_name = "dataset3"
number_of_epochs = 3
batch_sequence_size = 200
debugged_seq = 4

if generate_dataset:
    dataset = DU.gen_dataset(50000, 10, 5)
    DU.save_dataset(DATASET_PATH, dataset_name, dataset)

dataset = DU.load_dataset(DATASET_PATH, dataset_name)
tx, ty, tctr, vx, vy, vctr = DU.split_and_reshape_dataset(dataset, 0.8)
tends = DU.get_sequence_ends(tctr)
vends = DU.get_sequence_ends(vctr)
mod, opt = create_model2(5, 100, 10, learning_rate=learning_rate)
reporter = Reporter()
observation = {}
reporter.add_observer('main', mod)
reporter.add_observer('ntm', mod.ntm)

if load_model or not train:
    serializers.load_npz(MODEL_PATH + JNR + model_name + '.model', mod)
    serializers.load_npz(MODEL_PATH + JNR + model_name + '.opt', opt)

if train:
    val_losses = []
    train_losses = []
    prev_start = 0
    if debug_train:
        debug_observations = []
    seq_cou = 0
    for ep in range(number_of_epochs):
        if (ep + 1) * batch_sequence_size - 1 > tends.shape[0]:
            seq_cou = 0
        start_id = tends[seq_cou * batch_sequence_size]
        end_id = tends[(seq_cou + 1) * batch_sequence_size - 1]
        print "[%d,%d]" % (start_id, end_id)
        seq_cou += 1

        arr = LU.basic_train_model(mod, opt, (tx[start_id:end_id, :], ty[start_id:end_id, :], tctr[start_id:end_id, :]))
        loss = np.sum(np.asarray(arr)) / len(arr)
        val_arr = LU.basic_eval(mod, (vx[:500, :], vy[:500, :], vctr[:500, :]))
        val_loss = np.sum(np.asarray(val_arr)) / len(val_arr)
        if debug_train:
            first_seq_end = vends[debugged_seq - 1] + 1
            last_seq_end = vends[debugged_seq] + 1
            obs_tmp = RP.collect_observations(mod, reporter, vx[first_seq_end:last_seq_end],
                                              vctr[first_seq_end:last_seq_end],
                                              vy[first_seq_end:last_seq_end])
            debug_observations.append(obs_tmp)
        print "trn>" + str(loss) + " val>" + str(val_loss) + "; " + str(ep + 1) + "/" + str(number_of_epochs)
        val_losses.append(val_loss)
        train_losses.append(loss)

    serializers.save_npz(MODEL_PATH + JNR + model_name + '.model', mod)
    serializers.save_npz(MODEL_PATH + JNR + model_name + '.opt', opt)
    np.savetxt(LOSSES_PATH + JNR + TIMESTAMP + "_" + model_name + '_trn.csv', np.array(train_losses), fmt='%10.5f')
    np.savetxt(LOSSES_PATH + JNR + TIMESTAMP + "_" + model_name + '_val.csv', np.array(val_losses), fmt='%10.5f')

    if debug_train:
        debug_observation = RP.merge_observations(debug_observations)
        RP.save_observations(OBSERVATIONS_PATH, RP.gen_file_name(["debug", model_name, TIMESTAMP], "hdf5"),
                             debug_observation)
        pp = PdfPages(RP.join(PARAMETER_VIS_PATH, RP.gen_file_name(["debug", model_name, TIMESTAMP], "pdf")))
        RP.generate_ntm_control_vector_overview(debug_observation, pdf_file=pp)
        pp.close()

if show_last_losses:
    file_names = map(RP.parse_file_name, RP.get_dir_filenames(LOSSES_PATH))
    trns = filter(lambda x: x[0][-1] == "trn" and x[0][2] in model_name, file_names)
    max_trn = max(trns, key=lambda x: x[0][0])
    trn_np = RP.open_csvs_into_one(LOSSES_PATH, [RP.gen_file_name(*max_trn)])
    max_trn[0][-1] = "val"
    val_np = RP.open_csvs_into_one(LOSSES_PATH, [RP.gen_file_name(*max_trn)])
    pp = PdfPages(RP.join(LOSSES_PATH, RP.gen_file_name(["trnval", model_name, max_trn[0][0]], "pdf")))
    plt.plot(trn_np, "r-", val_np, "b-")
    pp.savefig()
    pp.close()

if collect_observations:
    last = vends[50] + 1
    observations = RP.collect_observations(mod, reporter, vx[:last, :], vctr[:last, :], t=vy[:last, :])
    RP.save_observations(OBSERVATIONS_PATH, RP.gen_file_name(["obs", model_name, TIMESTAMP], "hdf5"), observations)

if show_pics:
    observations = RP.load_observations(OBSERVATIONS_PATH, RP.gen_file_name(["obs", model_name, TIMESTAMP], "hdf5"))
    pp = PdfPages(RP.join(PARAMETER_VIS_PATH, RP.gen_file_name(['aftertrain', model_name, TIMESTAMP], 'pdf')))
    RP.generate_ntm_control_vector_overview(observations, number_of_sequences=10, pdf_file=pp)
    pp.close()
