import numpy as np
import matplotlib.pyplot as plt
from metrics import *
import os
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

data_type = "ood"
file_path = os.path.join(data_type)

image = np.load(os.path.join(file_path, "test_image_{}.npy".format(data_type)))
gt_one_hot = np.load(os.path.join(file_path, "test_mask.npy"))

prob_one_hot = np.load(os.path.join(file_path, "probs_mcmc_{}.npy".format(data_type)))
pred = np.load(os.path.join(file_path, "preds_mcmc_{}.npy".format(data_type)))
acc = np.load(os.path.join(file_path, "accs_mcmc_{}.npy").format(data_type))
conf = np.load(os.path.join(file_path, "confs_mcmc_{}.npy".format(data_type)))

print(np.unique(pred))

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


def compute_entropy():
    entropy = Entropy(prob_one_hot)
    # print(entropy.shape)
    np.save(os.path.join(file_path, "entropy_{}.npy".format(data_type)), entropy)
    return entropy


def compute_cdf():
    threshold = [0.0, 0.20, 0.50, 0.80, 1.09]

    entropy = compute_entropy()
    entropy_f = entropy.flatten()
    print(np.max(entropy_f))

    overlay = np.logical_or(gt_f, pred_f).astype(float)
    union = np.multiply(entropy_f, overlay).astype(float)
    union_without_zero = union[np.where(union != 0)]

    cdf_list = []

    for i in range(len(threshold)):
        cdf = (union_without_zero <= threshold[i]).sum() / union_without_zero.shape
        cdf_list.append(*cdf.round(5))
    print("cdf_{} = ".format(data_type), cdf_list)
    return cdf_list


def plot_cdf_entropy():
    threshold = [0.0, 0.20, 0.50, 0.80, "1.09"]
    cdf = compute_cdf()
    print("cdf_{} = ".format(data_type), cdf)

    fig = plt.figure()
    plt.plot(threshold, cdf)
    plt.title("CDF: mcmc {}".format(data_type))
    plt.xlabel("entropy")
    plt.ylabel("cdf")
    plt.show()
    fig.savefig(os.path.join(file_path, "cdf_kits_mcmc_{}.jpg".format(data_type)))


def plot_pdf_entropy():
    entropy = compute_entropy()
    entropy_f = entropy.flatten()

    overlay = np.logical_or(gt_f, pred_f).astype(float)
    union = np.multiply(entropy_f, overlay).astype(float)
    union_without_zero = union[np.where(union != 0)]

    fig = plt.figure()
    sns.histplot(union_without_zero, kde=True)
    plt.xlabel("entropy")
    plt.title("Density Plot: mcd")
    plt.show()
    fig.savefig(os.path.join(file_path, "pdf_kits_mcmc.jpg"))
    np.save(os.path.join(file_path, "pdf_kits_mcmc.npy".format(data_type)), union_without_zero)


def plot_uncertainty():
    font_size = 20
    image_index = [11, 20, 22, 35]
    slice_number = 64
    imgs = img[image_index]
    gts = gt[image_index]
    preds = pred[image_index]

    entropy = compute_entropy()
    ents = entropy[image_index]

    fig, axes = plt.subplots(len(imgs), 4, figsize=(12, 12))

    for i in range(len(imgs)):
        axes[i, 0].imshow(imgs[i, slice_number, :, :])
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title('input image', {'fontsize': font_size}, fontweight='bold')

        axes[i, 1].imshow(gts[i, slice_number, :, :])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        axes[i, 1].set_title('Ground Truth', {'fontsize': font_size}, fontweight='bold')

        axes[i, 2].imshow(preds[i, slice_number, :, :])
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        axes[i, 2].set_title('Pred mask', {'fontsize': font_size}, fontweight='bold')

        a1 = axes[i, 3].imshow(ents[i, slice_number, :, :])
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])
        axes[i, 3].set_title('Model Uncertainty', {'fontsize': font_size}, fontweight='bold')

        ax1_divider = make_axes_locatable(axes[i][3])
        cax1 = ax1_divider.append_axes("right", size="7%", pad=0.05)
        fig.colorbar(a1, cax=cax1)
    fig.tight_layout()
    fig.savefig(os.path.join(file_path, "uncertainty_plot_kits_mcmc_{}.jpg".format(data_type)))


def plot_slices():
    x = image[5].squeeze()
    plt.imshow(x[16, :, :])
    plt.show()


if __name__ == '__main__':
    # segmentation_metrics()
    # calibration_metrics()
    # plot_reliability_diagram()
    # compute_entropy()
    compute_cdf()
    plot_cdf_entropy()
    plot_pdf_entropy()
    plot_uncertainty()
    # plot_slices()
