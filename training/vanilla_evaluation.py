import os

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import utils

from torch_loader import *
from unet3D import Unet3D

device = utils.set_device()


def load_model(model_name, experiment, checkpoint):
    model = Unet3D(n_channels=1, n_classes=3, n_filters=32, drop=None, bilinear=True)
    weights_path = os.path.join('./trained_models', model_name, experiment, '{}.pt'.format(checkpoint))
    model = utils.load_weights(model, weights_path)
    return model


@torch.no_grad()
def predict_masks(model_name='vanilla', experiment='model_2',
                  checkpoint='final', data_type="ood"):  # Change the data_type

    model = load_model(model_name=model_name, experiment=experiment, checkpoint=checkpoint)

    outpath = os.path.join('./trained_models', model_name, experiment, "results", data_type)
    utils.mdir(outpath)

    model.to(device)

    probs_list = []
    confs_list = []
    preds_list = []
    accs_list = []

    # test = test_data_loader()  # same distribution dataset
    test = test_data_loader_ood()  # OOD Dataset

    for idx, test_batch in tqdm(enumerate(test)):
        inputs, labels = test_batch  # numpy data

        inputs, targets = inputs.to(device), labels.to(device)

        out = model(inputs)

        probs = F.softmax(out, dim=1)

        confs, preds = probs.max(dim=1, keepdim=False)

        gt = targets.argmax(1)
        accs = preds.eq(gt)

        probs = probs.cpu().numpy()
        confs = confs.cpu().numpy()
        preds = preds.cpu().numpy()
        accs = accs.cpu().numpy()

        probs_list.append(probs)
        confs_list.append(confs)
        preds_list.append(preds)
        accs_list.append(accs)

    probs = np.concatenate(probs_list, axis=0)
    confs = np.concatenate(confs_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    accs = np.concatenate(accs_list, axis=0)

    print(probs.shape, confs.shape, preds.shape, accs.shape)

    name = "{}_{}_{}".format(model_name, experiment, data_type)
    np.save(os.path.join(outpath, 'probs_{}.npy'.format(name)), probs)
    np.save(os.path.join(outpath, 'confs_{}.npy'.format(name)), confs)
    np.save(os.path.join(outpath, 'preds_{}.npy'.format(name)), preds)
    np.save(os.path.join(outpath, 'accs_{}.npy'.format(name)), accs)

    print(np.unique(preds))

    print("evaluate model_.{}".format(name))


if __name__ == '__main__':
    print("evaluate model")
    predict_masks()
