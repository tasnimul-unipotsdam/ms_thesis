import os

import torch
import numpy as np
from tqdm import tqdm

import utils

from torch_loader import torch_data_loader
from unet3D import Unet3D
from test_data_loader import torch_test_data_loader

device = utils.set_device()

drop_rate = [0.05, 0.1, 0.2, 0.3]


def load_model(model_name, experiment, checkpoint='final'):
    model = Unet3D(n_channels=1, n_classes=1, n_filters=16, drop=drop_rate[1])
    weights_path = os.path.join('./trained_models_brats', model_name, experiment, '{}.pt'.format(checkpoint))
    model = utils.load_weights(model, weights_path)
    return model


@torch.no_grad()
def predict_masks(model_name='ensemble', experiment='en_exp_13', checkpoint='final'):  # Change pred_name at line 47
    model = load_model(model_name=model_name, experiment=experiment, checkpoint=checkpoint)

    mask_outpath = os.path.join('./trained_models_brats', "ensemble", "ood")
    utils.mdir(mask_outpath)

    model.to(device)
    outputs = []

    # _, _, test = torch_data_loader()  # same distribution dataset
    test = torch_test_data_loader()  # OOD dataset

    for idx, test_batch in tqdm(enumerate(test)):
        inputs, labels = test_batch
        inputs, targets = inputs.to(device), labels.to(device)

        pred_mask = model(inputs)
        pred_mask = torch.sigmoid(pred_mask).cpu().numpy()
        outputs.append(pred_mask)

    outputs = np.concatenate(outputs, axis=0)

    pred_name = "pred_{}_ood".format(experiment)
    np.save(os.path.join(mask_outpath, '{}.npy'.format(pred_name)), outputs)
    print("evaluate model_.{}".format(pred_name))
    return outputs


def predict_masks_en(pred_name="pred_mcd_0.1_same_avg"):
    t = 20
    mask = predict_masks()
    for i in range(t - 1):
        print('model', i + 1, 'of', t - 1)
        mask = mask + predict_masks()

    print("taking average")
    Y_ts_hat = mask / t

    print("saving")
    mask_outpath = os.path.join('./trained_models_brats', "mc_drop", "same_dist")
    utils.mdir(mask_outpath)
    np.save(os.path.join(mask_outpath, '{}.npy'.format(pred_name)), Y_ts_hat)


def average_models(n_models=20):
    path = './trained_models_brats/ensemble/same_dist'

    print("load")

    model_01 = np.load(os.path.join(path, "pred_en_exp_13_same.npy"))
    model_02 = np.load(os.path.join(path, "pred_en_exp_21_same.npy"))
    model_03 = np.load(os.path.join(path, "pred_en_exp_22_same.npy"))
    model_04 = np.load(os.path.join(path, "pred_en_exp_35_same.npy"))
    model_05 = np.load(os.path.join(path, "pred_en_exp_40_same.npy"))
    model_06 = np.load(os.path.join(path, "pred_en_exp_49_same.npy"))
    model_07 = np.load(os.path.join(path, "pred_en_exp_50_same.npy"))
    model_08 = np.load(os.path.join(path, "pred_en_exp_53_same.npy"))
    model_09 = np.load(os.path.join(path, "pred_en_exp_56_same.npy"))
    model_10 = np.load(os.path.join(path, "pred_en_exp_59_same.npy"))
    model_11 = np.load(os.path.join(path, "pred_en_exp_63_same.npy"))
    model_12 = np.load(os.path.join(path, "pred_en_exp_69_same.npy"))
    model_13 = np.load(os.path.join(path, "pred_en_exp_73_same.npy"))
    model_14 = np.load(os.path.join(path, "pred_en_exp_77_same.npy"))
    model_15 = np.load(os.path.join(path, "pred_en_exp_78_same.npy"))
    model_16 = np.load(os.path.join(path, "pred_en_exp_83_same.npy"))
    model_17 = np.load(os.path.join(path, "pred_en_exp_86_same.npy"))
    model_18 = np.load(os.path.join(path, "pred_en_exp_90_same.npy"))
    model_19 = np.load(os.path.join(path, "pred_en_exp_92_same.npy"))
    model_20 = np.load(os.path.join(path, "pred_en_exp_98_same.npy"))

    print("average")

    average = (model_01 + model_02 + model_03 + model_04 + model_05 + model_06 + model_07 + model_08 + model_09 +
               model_10 + model_11 + model_12 + model_13 + model_14 + model_15 + model_16 + model_17 + model_18 +
               model_19 + model_20) / n_models

    avg_model_path = './trained_models_brats/ensemble/avg_model'
    np.save(os.path.join(avg_model_path, 'pred_en_avg20_same.npy'), average)


if __name__ == '__main__':
    print("evaluate model")
    # predict_masks()
    # predict_masks_en()
    average_models()
