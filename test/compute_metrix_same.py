import numpy as np
import matplotlib.pyplot as plt
from metrics import *
import os
import seaborn as sns

data_type = "same"

file_path = os.path.join(data_type)

image = np.load(os.path.join(file_path, "test_image.npy"))
gt_one_hot = np.load(os.path.join(file_path, "test_mask.npy"))

prob_one_hot = np.load(os.path.join(file_path, "probs_mcmc_{}.npy".format(data_type)))
pred = np.load(os.path.join(file_path, "preds_mcmc_{}.npy".format(data_type)))
acc = np.load(os.path.join(file_path, "accs_mcmc_{}.npy").format(data_type))
conf = np.load(os.path.join(file_path, "confs_mcmc_{}.npy".format(data_type)))

img = image.squeeze()
gt = gt_one_hot.argmax(1)

conf_f = conf.flatten()
acc_f = acc.flatten()
pred_f = pred.flatten()
gt_f = gt.flatten()


def segmentation_metrics():
    dice = []
    for i in range(len(pred)):
        dice.append(dice_coefficient(pred[i], gt[i]))
    dice = np.array(dice)
    dice_mean = np.mean(dice)
    metrics = {"Mean of dice coefficient": dice_mean}
    print(metrics)


def calibration_metrics():
    score = []
    for i in range(len(gt_one_hot)):
        score.append(brier_multi(gt_one_hot[i], prob_one_hot[i]))
    score = np.array(score)
    mean_score = np.mean(score)

    nll_score = []
    for i in range(len(gt_one_hot)):
        nll_score.append(nll_multi(gt_one_hot[i], prob_one_hot[i]))
    nll = np.array(nll_score)
    mean_nll = np.mean(nll)

    ece, _, _ = ECE(conf_f, acc_f, n_bins=10)

    metrics = {"Mean of brier score": mean_score,
               "mean of nll score": mean_nll,
               "expected calibration error": ece
               }
    print(metrics)


def plot_reliability_diagram():
    _, acc_list, conf_list = ECE(conf_f, acc_f, n_bins=10)
    print("acc_list_{} = ".format(data_type), acc_list)
    print("conf_list_{} = ".format(data_type), conf_list)
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(acc_list, conf_list, marker=".")
    plt.title("RD: mcmc {}".format(data_type))
    plt.show()
    fig.savefig(os.path.join(file_path, "RD_kits_mcmc_{}.jpg".format(data_type)))


def plot_prediction():
    font_size = 20
    image_index = [2, 4, 6, 8, 9]
    slice_number = 64
    imgs = img[image_index]
    gts = gt[image_index]
    preds = pred[image_index]

    sns.set_theme()

    fig, axes = plt.subplots(len(imgs), 3, figsize=(25, 25))

    for i in range(len(imgs)):
        axes[i, 0].imshow(imgs[i, :, slice_number, :])
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title('input image', {'fontsize': font_size}, fontweight='bold')

        axes[i, 1].imshow(gts[i, :, slice_number, :])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        axes[i, 1].set_title('Ground Truth', {'fontsize': font_size}, fontweight='bold')

        axes[i, 2].imshow(preds[i, :, slice_number, :])
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        axes[i, 2].set_title('Pred mask', {'fontsize': font_size}, fontweight='bold')

    fig.show()
    fig.savefig(os.path.join(file_path, "pred_plot_amos_mcmc_{}.jpg".format(data_type)))


if __name__ == '__main__':
    segmentation_metrics()
    calibration_metrics()
    plot_reliability_diagram()
    # plot_prediction()
