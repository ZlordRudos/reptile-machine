import numpy as np
from chainer import Variable
from dataset_utils import is_dataset_type, NOT_DATASET_MESSAGE, is_learn_on, is_sequence_end


def basic_eval(model, validation_set):
    assert is_dataset_type(validation_set), NOT_DATASET_MESSAGE

    ret_arry = []
    inpu, targ, ctr = validation_set
    model.reset()
    for i in range(inpu.shape[0]):
        err = model(Variable(inpu[i, :], volatile=True), Variable(targ[i, :], volatile=True))
        if is_learn_on(ctr[i, :]):
            ret_arry.append(err.data)
        if is_sequence_end(ctr[i, :]):
            model.reset()
    return ret_arry


def basic_train_model(model, optimizer, training_set, clip_grads=None):
    assert is_dataset_type(training_set), NOT_DATASET_MESSAGE

    accum_loss = Variable(np.zeros((), dtype=np.float32), volatile=False)
    ret_arr = []
    inpu, targ, ctr = training_set
    model.reset()
    for i in range(inpu.shape[0]):
        err = model(Variable(inpu[i, :], volatile=False), Variable(targ[i, :], volatile=False))
        if is_learn_on(ctr[i, :]):
            accum_loss += err
            ret_arr.append(err.data)
        if is_sequence_end(ctr[i, :]):
            model.reset()
    optimizer.zero_grads()
    accum_loss.backward()
    if clip_grads is not None:
        optimizer.clip_grads(clip_grads)
    optimizer.update()
    return ret_arr


def online_train_model(model, optimizer, training_set):
    assert is_dataset_type(training_set), NOT_DATASET_MESSAGE

    accum_loss = Variable(np.zeros((), dtype=np.float32), volatile=False)
    ret_arr = []
    sum_loss = np.zeros((), dtype=np.float32)
    inpu, targ, ctr = training_set
    prev_ctr = 1
    for i in range(inpu.shape[0]):
        if prev_ctr == 1 and ctr[i, 0] == 0:
            model.reset()
        prev_ctr = ctr[i, 0]
        err = model(Variable(inpu[i, :], volatile=False), Variable(targ[i, :], volatile=False))
        if ctr[i, 0] == 1:
            ret_arr.append(err.data)
            accum_loss += err
        if (i == inpu.shape[0] - 1) or (prev_ctr == 1 and ctr[i + 1, 0] == 0):
            optimizer.zero_grads()
            accum_loss.backward()
            optimizer.clip_grads(100)
            optimizer.update()
            sum_loss += accum_loss.data
            accum_loss = Variable(np.zeros((), dtype=np.float32), volatile=False)
    return ret_arr
