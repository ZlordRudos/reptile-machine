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
from ntm_wrapper import get_sections_head_controls, NeuralTuringMachineWrapper

MODEL_NAME = "demo_two_heads_copy"

MODEL_PATH = RP.join(RP.ROOT_MODEL_PATH, MODEL_NAME)
RESULTS_PATH = RP.join(MODEL_PATH, "results")
PARAMETER_VIS_PATH = RP.join(RESULTS_PATH, "parameter_vis")
LOSSES_PATH = RP.join(RESULTS_PATH, "losses")
OBSERVATIONS_PATH = RP.join(RESULTS_PATH, "observations")
DATASET_PATH = RP.join("resources", "datasets")

TIMESTAMP = datetime.datetime.now().strftime("%y%m%d%H%M")

RP.check_and_make_dirs(MODEL_PATH,
                       PARAMETER_VIS_PATH,
                       LOSSES_PATH,
                       OBSERVATIONS_PATH
                       )


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


def create_model(element_size, hidden_size, memory_size, max_shift, learning_rate=10e-5):
    head_order = ["w", "r"]
    controller = ReluForwardController(element_size, hidden_size, max_shift, head_order=head_order)
    wrapper = NeuralTuringMachineWrapper(controller, memory_size, element_size, max_shift, head_order)
    model = MSEError(wrapper)
    optimizer = optimizers.RMSpropGraves()
    optimizer.setup(model)
    optimizer.lr = learning_rate
    return model, optimizer


####################################################
####################################################

lr = 10e-4
model_name = "demo1"
dataset_name = "dataset3"
number_of_epochs = 100
batch_sequence_size = 200
debugged_seq = 4

####################################################
####################################################

load_model = True  # False => creates/rewrites model with defined model_name
generate_dataset = False  # True => creates/rewrites dataset with defined dataset_name
show_loss_curves = False
debug_train = True  # True => shows progression of learning on one example sequence
train = False
# train = True
show_pics = True
# show_pics = False

####################################################
####################################################

if generate_dataset:
    dataset = DU.gen_dataset(50000, 10, 5)
    DU.save_dataset(DATASET_PATH, dataset_name, dataset)

dataset = DU.load_dataset(DATASET_PATH, dataset_name)
tx, ty, tctr, vx, vy, vctr = DU.split_and_reshape_dataset(dataset, 0.8)
tends = DU.get_sequence_ends(tctr)
vends = DU.get_sequence_ends(vctr)
mod, opt = create_model(5, 100, 10, 1, learning_rate=lr)
reporter = Reporter()
observation = {}
reporter.add_observer('main', mod)
reporter.add_observer('ntm', mod.ntm.heads[0])
reporter.add_observer('ntm_r', mod.ntm.heads[1])
model_path = RP.join(MODEL_PATH, RP.gen_file_name(model_name, 'model'))
optimizer_path = RP.join(MODEL_PATH, RP.gen_file_name(model_name, 'opt'))

if load_model or not train:
    serializers.load_npz(model_path, mod)
    serializers.load_npz(optimizer_path, opt)

if train:
    # Init arrays for loss function recording
    val_losses = []
    train_losses = []
    prev_start = 0
    seq_cou = 0
    if debug_train:
        # Init array for observation collections
        debug_observations = []
    for ep in range(number_of_epochs):
        # Creating batch with unbroken first and last sequences
        if (ep + 1) * batch_sequence_size - 1 > tends.shape[0]:
            seq_cou = 0
        start_id = tends[seq_cou * batch_sequence_size]
        end_id = tends[(seq_cou + 1) * batch_sequence_size - 1]
        print "[%d,%d]" % (start_id, end_id)
        seq_cou += 1
        # Training
        arr = LU.basic_train_model(mod, opt, (tx[start_id:end_id, :], ty[start_id:end_id, :], tctr[start_id:end_id, :]))
        # Collecting training and validation loss
        loss = np.sum(np.asarray(arr)) / len(arr)
        val_arr = LU.basic_eval(mod, (vx[:500, :], vy[:500, :], vctr[:500, :]))
        val_loss = np.sum(np.asarray(val_arr)) / len(val_arr)
        print "trn>" + str(loss) + " val>" + str(val_loss) + "; " + str(ep + 1) + "/" + str(number_of_epochs)
        val_losses.append(val_loss)
        train_losses.append(loss)
        if debug_train:
            # Selecting validation sequence
            first_seq_end = vends[debugged_seq - 1] + 1
            last_seq_end = vends[debugged_seq] + 1
            # Creating new observations collection
            obs_tmp = RP.collect_observations(mod, reporter, vx[first_seq_end:last_seq_end],
                                              vctr[first_seq_end:last_seq_end],
                                              vy[first_seq_end:last_seq_end])
            debug_observations.append(obs_tmp)
    # Save model and optimizer state
    serializers.save_npz(model_path, mod)
    serializers.save_npz(optimizer_path, opt)
    # Save collected validation and training loss
    np.savetxt(RP.join(LOSSES_PATH, RP.gen_file_name(model_name, 'trn', TIMESTAMP, 'csv')),
               np.array(train_losses), fmt='%10.5f')
    np.savetxt(RP.join(LOSSES_PATH, RP.gen_file_name(model_name, 'val', TIMESTAMP, 'csv')),
               np.array(val_losses), fmt='%10.5f')

    if debug_train:
        # Save observation collections and generate pdf output
        debug_observation = RP.merge_observations(debug_observations)
        RP.save_observations(OBSERVATIONS_PATH, RP.gen_file_name(model_name, "debug", TIMESTAMP, "hdf5"),
                             debug_observation)
        pp = PdfPages(RP.join(PARAMETER_VIS_PATH, RP.gen_file_name(model_name, "debug", TIMESTAMP, "pdf")))
        RP.generate_ntm_control_vector_overview(debug_observation, pdf_file=pp)
        pp.close()

    # Create pdf output of validation and training loss
    pp = PdfPages(RP.join(LOSSES_PATH, RP.gen_file_name("trnval", model_name, TIMESTAMP, "pdf")))
    plt.plot(np.array(train_losses), "r-", np.array(val_losses), "b-")
    pp.savefig()
    pp.close()

# Aggregate all previous losses of this model into plot
if show_loss_curves:
    file_names = map(RP.parse_file_name, RP.get_dir_filenames(LOSSES_PATH))
    # Filter by model
    trns = filter(lambda x: x[0][1] == "trn" and x[0][0] in model_name, file_names)
    # Sort by stamp
    max_trn = sorted(trns, key=lambda x: x[0][-1])
    # Open training loss into one np.array
    trn_np = RP.open_csvs_into_one(LOSSES_PATH, map(RP.gen_file_name, max_trn))
    max_trn[:][0][1] = 'val'
    # Open validation loss into one np.array
    val_np = RP.open_csvs_into_one(LOSSES_PATH, map(RP.gen_file_name, max_trn))
    # Create and save the plot
    pp = PdfPages(RP.join(LOSSES_PATH, RP.gen_file_name(model_name, "trnval_all", max_trn[0][0][-1], "pdf")))
    plt.plot(trn_np, "r-", val_np, "b-")
    pp.savefig()
    pp.close()

# Generate portfolio of various sequences
if show_pics:
    last = vends[50] + 1
    observations = RP.collect_observations(mod, reporter, vx[:last, :], vctr[:last, :], t=vy[:last, :])
    RP.save_observations(OBSERVATIONS_PATH, RP.gen_file_name(model_name, "obs", TIMESTAMP, "hdf5"), observations)
    observations = RP.load_observations(OBSERVATIONS_PATH, RP.gen_file_name(model_name, "obs", TIMESTAMP, "hdf5"))
    pp = PdfPages(RP.join(PARAMETER_VIS_PATH, RP.gen_file_name(model_name, 'aftertrain', TIMESTAMP, 'pdf')))
    RP.generate_ntm_control_vector_overview(observations, number_of_sequences=50, pdf_file=pp)
    pp.close()
