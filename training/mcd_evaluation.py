import os

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import utils

from torch_loader import *
from unet3D import Unet3D

device = utils.set_device()

drop_rate = [0.01, 0.05, 0.1, 0.2, 0.3]


def load_model(model_name, experiment, checkpoint):
    model = Unet3D(n_channels=1, n_classes=3, n_filters=32, drop=0.05, bilinear=True)
    weights_path = os.path.join('./trained_models', model_name, experiment, '{}.pt'.format(checkpoint))
    model = utils.load_weights(model, weights_path)
    return model


@torch.no_grad()
def predict_masks(model_name='mcd', experiment='model_2',
                  checkpoint='final'):
    model = load_model(model_name=model_name, experiment=experiment, checkpoint=checkpoint)
    model.to(device)

    # test = test_data_loader(batch_size=1)  # same distribution dataset
    test = test_data_loader_ood(batch_size=1)  # OOD Dataset

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
def predict_masks_mcdrop():  # Change the data_type
    t = 20
    prob = predict_masks()
    for i in range(t - 1):
        print('model', i + 1, 'of', t - 1)
        prob = prob + predict_masks()
    print("taking average")
    probs_avg = prob / t

    return probs_avg


def calculation(data_type="ood_0.05", model_name='mcd'):  # Change the data_type

    out_path = os.path.join('./trained_models', model_name, 'model_2', "results", data_type)
    utils.mdir(out_path)

    probs_avg = predict_masks_mcdrop()

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
