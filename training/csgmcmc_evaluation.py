import os
import glob
from glob import glob
import os

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import utils

from torch_loader import *
from unet3D import Unet3D

device = utils.set_device()


def paths():
    path = './/trained_models/mcmc/model_6'
    weight_path = sorted(glob(os.path.join(path, "checkpoint_*")))
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_50.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_100.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_150.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_200.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_250.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_300.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_350.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_400.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_450.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_500.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_550.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_600.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_650.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_700.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_750.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_800.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_850.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_900.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_950.pt')
    weight_path.remove('.//trained_models/mcmc/model_6/checkpoint_1000.pt')

    return weight_path


weight_path = paths()
print(len(weight_path))


def load_model(file_paths):
    model = Unet3D(n_channels=1, n_classes=3, n_filters=32, drop=None, bilinear=True)
    model = utils.load_weights(model, file_paths)
    return model


@torch.no_grad()
def predict_masks(file_paths=weight_path):
    model = load_model(file_paths)
    model.to(device)

    # test = test_data_loader()  # same distribution dataset
    test = test_data_loader_ood()  # OOD Dataset

    probs_list = []

    for idx, test_batch in tqdm(enumerate(test)):
        inputs, labels = test_batch

        inputs, targets = inputs.to(device), labels.to(device)

        out = model(inputs)

        probs = F.softmax(out, dim=1)
        probs = probs.cpu().numpy()
        probs_list.append(probs)

    probs = np.concatenate(probs_list, axis=0)

    return probs


@torch.no_grad()
def predict_masks_mcmc():  # Change the data_type

    prob = predict_masks(weight_path[0])
    print(weight_path[0])

    for i in range(len(weight_path) - 1):
        print('model', i + 1, 'of', len(weight_path) - 1)

        prob = prob + predict_masks(weight_path[i + 1])
        print(weight_path[i + 1])

    print("taking average")

    probs_avg = prob / len(weight_path)

    return probs_avg


def calculation(data_type="ood", model_name='mcmc'):  # Change the data_type

    out_path = os.path.join('./trained_models', model_name, "model_6", "mcmc_results", data_type)
    utils.mdir(out_path)

    probs_avg = predict_masks_mcmc()

    # test = test_data_loader(batch_size=len(probs_avg))  # same distribution dataset
    test = test_data_loader_ood(batch_size=len(probs_avg))  # OOD Dataset

    print("calculate all")

    for idx, test_batch in tqdm(enumerate(test)):
        inputs, labels = test_batch

        inputs, targets = inputs.to(device), labels.to(device)

        probs_avg = torch.from_numpy(probs_avg)
        probs = probs_avg.to(device)

        confs, preds = probs.max(dim=1, keepdim=False)
        gt = targets.argmax(1)
        accs = preds.eq(gt)

        probs = probs.cpu().numpy()
        confs = confs.cpu().numpy()
        preds = preds.cpu().numpy()
        accs = accs.cpu().numpy()

        print(probs.shape, confs.shape, preds.shape, accs.shape)

        print("saving")

        name = "{}_{}".format(model_name, data_type)
        np.save(os.path.join(out_path, 'probs_{}.npy'.format(name)), probs)
        np.save(os.path.join(out_path, 'confs_{}.npy'.format(name)), confs)
        np.save(os.path.join(out_path, 'preds_{}.npy'.format(name)), preds)
        np.save(os.path.join(out_path, 'accs_{}.npy'.format(name)), accs)

        print(np.unique(preds))

        print("evaluate model_.{}".format(name))


if __name__ == '__main__':
    print("evaluate model")
    calculation()