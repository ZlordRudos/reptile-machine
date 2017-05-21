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
from links.ntm_one_head import NtmOneHeadWrapper
from ntm_wrapper import get_sections_head_controls, NeuralTuringMachineWrapper

JNR = "/"
DATASET_PATH = "resources/datasets"
MODEL_PATH = "resources/models/exp1"
RESULTS_PATH = MODEL_PATH + JNR + "results"
PARAMETER_VIS_PATH = RESULTS_PATH + JNR + "parameter_vis"
LOSSES_PATH = RESULTS_PATH + JNR + "losses"
OBSERVATIONS_PATH = RESULTS_PATH + JNR + "observations"

TIMESTAMP = datetime.datetime.now().strftime("%y%m%d%H%M")

RP.check_and_make_dirs(MODEL_PATH,
                       PARAMETER_VIS_PATH,
                       LOSSES_PATH,
                       OBSERVATIONS_PATH
                       )

# TODO: Cteci a zapisovaci hlavy
# TODO: pamet jako vystup
# TODO: zihani
# TODO: pergamen vis
# TODO: Dynamicka pamet?

class ReluForwardController(chainer.Chain):
    def __init__(self, input_len, hidden_size, max_shift, head_order=["wr"]):
        super(ReluForwardController, self).__init__(
            l0=F.Linear(input_len * (sum(map(lambda x: 'r' in x, head_order)) + 1),
                        hidden_size),
            l1=F.Linear(hidden_size,
                        get_sections_head_controls(input_len, max_shift,
                                                   head_order)[-1]
                        + input_len))

    def __call__(self, x):
        h1 = F.tanh(self.l0(x))
        return F.tanh(self.l1(h1))


class MSEError(chainer.Chain):
    def __init__(self, ntm_wrapper):
        super(MSEError, self).__init__(ntm=ntm_wrapper)

    def reset(self):
        self.ntm.reset()

    def predict(self, inp):
        return self.ntm(inp)

    def __call__(self, inp, t):
        oup = self.predict(inp)
        return F.mean_squared_error(oup, t)


def create_model2(element_size, hidden_size, memory_size, max_shift, learning_rate=10e-5):
    model = MSEError(
        NtmOneHeadWrapper(ReluForwardController(element_size, hidden_size, max_shift), memory_size,
                          element_size,
                          max_shift))
    optimizer = optimizers.RMSpropGraves()
    optimizer.setup(model)
    optimizer.lr = learning_rate
    return model, optimizer


def create_model3(element_size, hidden_size, memory_size, max_shift, learning_rate=10e-5):
    controller = ReluForwardController(element_size, hidden_size, max_shift)
    wrapper = NeuralTuringMachineWrapper(controller, memory_size, element_size, max_shift, ["wr"])
    model = MSEError(wrapper)
    optimizer = optimizers.RMSpropGraves()
    optimizer.setup(model)
    optimizer.lr = learning_rate
    return model, optimizer


######################################################################
######################################################################
lr = 10e-4
train_type = 0  # 0 - basic
load_model = True
debug_train = True
train = False
# train = True
show_pics = True
# show_pics = False
generate_dataset = False
model_name = "test_relu6"
dataset_name = "dataset3"
number_of_epochs = 500
batch_sequence_size = 200
debugged_seq = 4

if generate_dataset:
    dataset = DU.gen_dataset(50000, 10, 5)
    DU.save_dataset(DATASET_PATH, dataset_name, dataset)

dataset = DU.load_dataset(DATASET_PATH, dataset_name)
tx, ty, tctr, vx, vy, vctr = DU.split_and_reshape_dataset(dataset, 0.8)
tends = DU.get_sequence_ends(tctr)
vends = DU.get_sequence_ends(vctr)
mod, opt = create_model3(5, 100, 10, 1, learning_rate=lr)
reporter = Reporter()
observation = {}
reporter.add_observer('main', mod)
reporter.add_observer('ntm', mod.ntm.heads[0])

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
        RP.save_observations(OBSERVATIONS_PATH, RP.gen_file_name("debug", model_name, TIMESTAMP, "hdf5"),
                             debug_observation)
        pp = PdfPages(RP.join(PARAMETER_VIS_PATH, RP.gen_file_name("debug", model_name, TIMESTAMP, "pdf")))
        RP.generate_ntm_control_vector_overview(debug_observation, pdf_file=pp)
        pp.close()

if train:
    file_names = map(RP.parse_file_name, RP.get_dir_filenames(LOSSES_PATH))
    trns = filter(lambda x: x[0][-1] == "trn" and x[0][2] in model_name, file_names)
    max_trn = max(trns, key=lambda x: x[0][0])
    trn_np = RP.open_csvs_into_one(LOSSES_PATH, [RP.gen_file_name(*max_trn)])
    max_trn[0][-1] = "val"
    val_np = RP.open_csvs_into_one(LOSSES_PATH, [RP.gen_file_name(*max_trn)])
    pp = PdfPages(RP.join(LOSSES_PATH, RP.gen_file_name("trnval", model_name, max_trn[0][0], "pdf")))
    plt.plot(trn_np, "r-", val_np, "b-")
    pp.savefig()
    pp.close()

if show_pics:
    last = vends[50] + 1
    observations = RP.collect_observations(mod, reporter, vx[:last, :], vctr[:last, :], t=vy[:last, :])
    RP.save_observations(OBSERVATIONS_PATH, RP.gen_file_name("obs", model_name, TIMESTAMP, "hdf5"), observations)
    observations = RP.load_observations(OBSERVATIONS_PATH, RP.gen_file_name("obs", model_name, TIMESTAMP, "hdf5"))
    pp = PdfPages(RP.join(PARAMETER_VIS_PATH, RP.gen_file_name('aftertrain', model_name, TIMESTAMP, 'pdf')))
    RP.generate_ntm_control_vector_overview(observations, number_of_sequences=50, pdf_file=pp)
    pp.close()
