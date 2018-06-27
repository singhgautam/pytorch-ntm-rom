import torch
import random
import numpy as np
from torch import optim
from modelcell import ModelCell
from tasks.copytask import CopyTaskParams
import logging
import json
import time
import torchvision


LOGGER = logging.getLogger(__name__)

# initialize CUDA
print 'torch.version',torch.__version__
print 'torch.cuda.is_available()',torch.cuda.is_available()
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True}

def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(time.time())

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def generate_random_batch(params):
    # All batches have the same sequence length
    seq_len = random.randint(params.sequence_min_len,
                             params.sequence_max_len)
    seq = np.random.binomial(1, 0.5, (seq_len,
                                      params.batch_size,
                                      params.sequence_width))
    seq = torch.from_numpy(seq).float()

    # The input includes an additional channel used for the delimiter
    inp = torch.zeros(seq_len + 1, params.batch_size, params.sequence_width + 1)
    inp[:seq_len, :, :params.sequence_width] = seq
    inp[seq_len, :, params.sequence_width] = 1.0  # delimiter in our control channel
    outp = seq.clone()

    return inp, outp

def clip_grads(model, range):
    """Gradient clipping to the range."""
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-range, range)

def batch_progress_bar(batch_num, report_interval, last_loss, bad_bits):
    """Prints the progress until the next report."""
    progress = (((batch_num - 1.0) % report_interval) + 1.0) / report_interval
    fill = int(progress * 40)
    print "\r\tBATCH [{}{}]: {} (Loss: {:.4f}) (Bad Bits: {:.4f})".format(
        "=" * fill,
        " " * (40 - fill),
        batch_num,
        last_loss,
        bad_bits)

def mean_progress(batch_num, mean_loss, mean_bad_bits):
    print "BATCH {} (Loss: {:.4f}) (Bad Bits: {:.4f})".format(
        batch_num,
        mean_loss,
        mean_bad_bits)

def save_checkpoint_for_batch(checkpoint_path,
                              model,
                              name,
                              seed,
                              batch_num,
                              losses,
                              costs,
                              seq_lengths):
    basename = "{}/{}-{}-batch-{}".format(checkpoint_path, name, seed, batch_num)
    model_fname = basename + ".model"
    LOGGER.info("Saving model checkpoint to: '%s'", model_fname)
    torch.save(model.state_dict(), model_fname)

    # Save the training history for batch
    train_fname = basename + ".json"
    LOGGER.info("Saving model training history to '%s'", train_fname)
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
    }
    open(train_fname, 'wt').write(json.dumps(content))


"""MAKE MODEL"""

init_seed(1000)

params = CopyTaskParams()

# init model cell
modelcell = ModelCell(params)
modelcell.memory.reset(params.batch_size)
modelcell.state.reset(params.batch_size)
modelcell.controller.reset_parameters()

print 'Memory is on CUDA : {}'.format(modelcell.memory.memory.is_cuda)

optimizer = optim.RMSprop(modelcell.parameters(),
                          momentum=params.rmsprop_momentum,
                          alpha=params.rmsprop_alpha,
                          lr=params.rmsprop_lr)

"""START TRAINING MODEL"""
loss_history = []
bad_bits_history = []
for batch_num in range(params.num_batches):
    # reset the states
    modelcell.memory.reset(params.batch_size)
    modelcell.state.reset(params.batch_size)

    # init optimizer
    optimizer.zero_grad()

    # generate data for the copy task
    X, Y = generate_random_batch(params)

    X.to(device)
    Y.to(device)

    # input phase
    for i in range(X.size(0)):
        _ = modelcell(X[i])

    # output phase
    Y_out = torch.zeros(Y.size())
    X_zero = torch.zeros(params.batch_size, params.sequence_width + 1)
    for i in range(Y.size(0)):
        Y_out[i] = modelcell(X_zero)[:,:params.sequence_width]

    loss = params.loss(Y_out, Y)
    loss_history.append(loss)

    loss.backward()
    clip_grads(modelcell, 10)
    optimizer.step()

    Y_out_binary = Y_out.clone().data
    Y_out_binary.apply_(lambda x: 0 if x < 0.5 else 1)

    bad_bits = torch.sum(torch.abs(Y_out_binary - Y_out))
    bad_bits_history.append(bad_bits)

    batch_progress_bar(batch_num+1, params.num_batches, last_loss=loss, bad_bits=bad_bits)
    if batch_num % params.save_every == 0 :
        mean_progress(batch_num,
                      sum(loss_history[-params.save_every:])/params.save_every,
                      sum(bad_bits_history[-params.save_every:])/params.save_every)

    if batch_num % params.illustrate_every == 0 :
        X, Y = params.get_illustrative_sample()

        modelcell.memory.reset(1)
        modelcell.state.reset(1)

        attention_history = torch.zeros(Y.size(0), modelcell.memory.N)

        # input phase
        for i in range(X.size(0)):
            _ = modelcell(X[i])

        # output phase
        Y_out = torch.zeros(Y.size())
        X_zero = torch.zeros(1, params.sequence_width + 1)
        for i in range(Y.size(0)):
            Y_out[i] = modelcell(X_zero)[:, :params.sequence_width]
            attention_history[i] = modelcell.state.readstate.w.squeeze()

        Y_out_binary = Y_out.clone().data
        Y_out_binary.apply_(lambda x: 0 if x < 0.5 else 1)

        # generate images
        torchvision.utils.save_image(X.squeeze(1), 'imsaves/illustrations/batch-{}-X.png'.format(batch_num))
        torchvision.utils.save_image(Y_out.squeeze(1), 'imsaves/illustrations/batch-{}-Y.png'.format(batch_num))
        torchvision.utils.save_image(attention_history, 'imsaves/illustrations/batch-{}-attention.png'.format(batch_num))
        torchvision.utils.save_image(Y_out_binary.squeeze(1), 'imsaves/illustrations/batch-{}-Y-binary.png'.format(batch_num))
        torchvision.utils.save_image(modelcell.memory.memory.squeeze(0), 'imsaves/illustrations/batch-{}-mem.png'.format(batch_num))