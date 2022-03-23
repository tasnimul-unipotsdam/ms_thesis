import os, glob
import numpy as np
import pandas as pd
import random
import torch
from PIL import Image
import torchvision
import torchvision.transforms as T
from collections import namedtuple
import copy
import argparse

import json

import torch
from torch import nn
from skimage.io import imread
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.io as spio
from scipy.ndimage import gaussian_filter

from torch import nn
import copy
import pandas as pd
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset
import nibabel as nib
from sklearn.metrics import auc

from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

from skimage.util import view_as_windows
from sklearn.metrics import confusion_matrix

import time

import datetime
import torch.utils.data
from torchvision import transforms, datasets
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.optim as optim

torch.manual_seed(42)
torch.cuda.manual_seed(42)

try:
    import cPickle as pickle
except:
    import pickle

from PIL import Image

import warnings

warnings.filterwarnings('ignore')


#####################


####################
# utilities
def one_hot(img, n_classes):
    h, w, d = img.shape

    one_hot = np.zeros((n_classes, h, w, d), dtype=np.float32)

    for i, unique_value in enumerate(np.unique(img)):
        one_hot[i, :, :, :][img == unique_value] = 1

    return (one_hot)


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        ## dump is used to write binary files
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def plotRocCurve(fdr, tpr, fpr, prec, svd):
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(fdr, tpr, lw=3)
    plt.xlabel('False Discovery Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('FDR vs TPR')
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, lw=3)
    plt.xlabel('False Positive Rate (Fall-Out)')
    plt.ylabel('True Positive Rate')
    plt.title('FPR vs TPR')
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(tpr, prec, lw=3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall vs Precision')
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

    fig.tight_layout(pad=3.0)

    plt.savefig(svd, bbox_inches='tight')
    plt.show()


def plotCurves(stats, titl, results_dir=None):
    fig = plt.figure(figsize=(12, 6))

    textsize = 15
    marker = 5

    plt.subplot(1, 2, 1)
    plt.plot(stats['train_loss'], label='train_loss')
    plt.plot(stats['valid_loss'], label='valid_loss')
    plt.xlabel('Epochs')
    plt.ylabel(f'{titl} Loss')
    plt.title('Loss Curve')
    plt.grid(True)
    lgd = plt.legend(['train loss', 'val loss'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})

    plt.subplot(1, 2, 2)
    plt.plot(stats['train_dice'], label='train_dice')
    plt.plot(stats['valid_dice'], label='valid_dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.title('Dice coefficient')
    plt.grid(True)
    lgd = plt.legend(['train', 'validation'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})

    fig.tight_layout(pad=3.0)
    if results_dir:
        plt.savefig(results_dir + 'cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.show()


def Unc_stats(unc):
    assert len(unc.shape) == 4
    min_unc = np.min(unc, axis=(1, 2, 3))
    max_unc = np.max(unc, axis=(1, 2, 3))
    mean_unc = np.mean(unc, axis=(1, 2, 3))

    return (min_unc, max_unc, mean_unc)


def Var_stats(var):
    """
       Here we look at varinace of the output!

    """

    assert len(var.shape) == 5
    min_var = np.min(var, axis=(1, 2, 3, 4))
    max_var = np.max(var, axis=(1, 2, 3, 4))
    mean_var = np.mean(var, axis=(1, 2, 3, 4))

    return (min_var, max_var, mean_var)


def visualize(path,
              imgs=None,
              title=None,
              cols=6,
              rows=1,
              plot_size=(12, 12),
              norm=False,
              slices=50,
              dts=None):
    fig, ax = plt.subplots(len(imgs), cols, figsize=plot_size)

    for j, img in enumerate(imgs):

        print(f'j:{j}')

        for i in range(cols):

            slice_img = img[i][:, :, slices]

            if j >= 3:

                if i == 0:
                    a0 = ax[j, i].imshow(slice_img, cmap=cm.coolwarm)

                ax[j, i].imshow(slice_img, cmap=cm.coolwarm)
            else:

                ax[j, i].imshow(slice_img)

            ax[j, i].set_title(f'{title[j]}')

            ax[j, i].axis('off')

        if j >= 3:
            ax1_divider = make_axes_locatable(ax[j, cols - 1])
            cax1 = ax1_divider.append_axes("right", size="7%", pad="4%")
            cbari = fig.colorbar(a0, ax=ax[j, cols - 1], cax=cax1)

    if path:
        fig.savefig(path)

    plt.show()
    plt.close()


def visualize3(path,
               imgs=None,
               title=None,
               cols=6,
               rows=1,
               plot_size=(12, 12),
               norm=False,
               slices=50,
               dts=None):
    """
       Here we plot img,gt,pred, unc in each row
    """
    img = imgs[0]
    gts = imgs[1]
    preds = imgs[2]
    uncs = imgs[3]

    rows = len(gts)
    cols = len(imgs)

    print(f'img shape in vis3: {img[0].shape}')

    prefix = ['', 'wt', 'tc', 'et']

    fig, ax = plt.subplots(rows, cols, figsize=plot_size)

    for i in range(rows):
        j = 0
        slice_img = img[i][0, :, :, slices]

        ax[i, j].imshow(slice_img)

        ax[i, j].set_title(f'image {title[0]}_slice {slices}', fontsize=10)

        ax[i, j].axis('off')

        j += 1
        slice_img = gts[i][:, :, slices]

        ax[i, j].imshow(slice_img)

        ax[i, j].set_title(f'gt-{prefix[i]} {title[0]}_slice {slices}', fontsize=10)

        ax[i, j].axis('off')

        j += 1

        slice_img = preds[i][:, :, slices]

        ax[i, j].imshow(slice_img)

        ax[i, j].set_title(f'pred-{prefix[i]} {title[0]}_slice {slices}', fontsize=10)

        ax[i, j].axis('off')

        j += 1

        slice_img = uncs[i][:, :, slices]

        a0 = ax[i, j].imshow(slice_img, cmap=cm.coolwarm)

        ax[i, j].set_title(f'unc_map_{prefix[i]} {title[0]}_slice {slices}', fontsize=10)

        ax[i, j].axis('off')

        ax1_divider = make_axes_locatable(ax[i, j])
        cax1 = ax1_divider.append_axes("right", size="7%", pad="4%")
        cbari = fig.colorbar(a0, ax=ax[i, j], cax=cax1)

    if path:
        fig.savefig(path)

    plt.show()
    plt.close()


def visualize2(path=None,
               imgs=None,
               title=None,
               cols=4,
               rows=2,
               plot_size=(8, 8),
               norm=False,
               slices=50,
               dts=None,
               unc_mod=None):
    fig, axes = plt.subplots(rows, cols, figsize=plot_size)

    print(f'img_shape: {imgs[0][0].shape}')
    # showing image
    slice_img = imgs[0][0][0, :, :, slices]
    axes[0, 0].imshow(slice_img)
    axes[0, 0].axis('off')
    axes[0, 0].set_title(f'img{title[0]}-slice{slices}', fontsize=22)
    # showing gt
    slice_img = imgs[1][0][:, :, slices]
    axes[0, 1].imshow(slice_img)
    axes[0, 1].axis('off')
    axes[0, 1].set_title(f'mask{title[0]}-slice{slices}', fontsize=22)
    # showing predicted mask
    slice_img = imgs[2][0][:, :, slices]
    axes[0, 2].imshow(slice_img)
    axes[0, 2].axis('off')
    axes[0, 2].set_title(f'prediction{title[0]}-slice{slices}', fontsize=22)
    # showing uncertainty map
    slice_img = imgs[3][0][:, :, slices]
    a3 = axes[0, 3].imshow(slice_img, cmap=cm.coolwarm)
    axes[0, 3].axis('off')
    axes[0, 3].set_title(f'unc_map{title[0]}-slice{slices}', fontsize=22)
    ax0_divider = make_axes_locatable(axes[0, 3])
    cax0 = ax0_divider.append_axes("right", size="7%", pad=0.07)
    cbar1 = fig.colorbar(a3, ax=axes[0, 3], cax=cax0)

    # showing img
    slice_img = imgs[0][1][0, :, :, slices]
    axes[1, 0].imshow(slice_img)
    axes[1, 0].axis('off')
    axes[1, 0].set_title(f'img{title[1]}-slice{slices}', fontsize=22)
    # showing gt
    slice_img = imgs[1][1][:, :, slices]
    axes[1, 1].imshow(slice_img)
    axes[1, 1].axis('off')
    axes[1, 1].set_title(f'mask{title[1]}-slice{slices}', fontsize=22)
    # showing predicted mask
    slice_img = imgs[2][1][:, :, slices]
    axes[1, 2].imshow(slice_img)
    axes[1, 2].axis('off')
    axes[1, 2].set_title(f'prediction{title[1]}-slice{slices}', fontsize=22)
    # showing unc_maps
    slice_img = imgs[3][1][:, :, slices]
    a3 = axes[1, 3].imshow(slice_img, cmap=cm.coolwarm)
    axes[1, 3].axis('off')
    axes[1, 3].set_title(f'unc_map{title[1]}-slice{slices}', fontsize=22)
    ax0_divider = make_axes_locatable(axes[1, 3])
    cax0 = ax0_divider.append_axes("right", size="7%", pad=0.07)
    cbar1 = fig.colorbar(a3, ax=axes[1, 3], cax=cax0)

    if unc_mod:
        print(f'unc_mod in vis2: {unc_mod}')

        # showing whole tumor
        slice_wt_gt = imgs[4][0][:, :, slices]
        axes[2, 0].imshow(slice_wt_gt)
        axes[2, 0].axis('off')
        axes[2, 0].set_title(f'Whole Tomur{title[0]}-slice{slices}', fontsize=22)
        # showing wohle tomur
        slice_wt_gt = imgs[4][0][:, :, slices]
        axes[2, 1].imshow(slice_wt_gt)
        axes[2, 1].axis('off')
        axes[2, 1].set_title(f'Whole Tomur{title[0]}-slice{slices}', fontsize=22)
        # showing wohle tomur pred
        slice_wt_prd = imgs[5][0][:, :, slices]
        axes[2, 2].imshow(slice_wt_prd)
        axes[2, 2].axis('off')
        axes[2, 2].set_title(f'prediction{title[0]}-slice{slices}', fontsize=22)
        # showing wt unc map
        slice_wt_unc = imgs[6][0][:, :, slices]
        a3 = axes[2, 3].imshow(slice_wt_unc, cmap=cm.coolwarm)
        axes[2, 3].axis('off')
        axes[2, 3].set_title(f'unc_map{title[0]}-slice{slices}', fontsize=22)
        ax0_divider = make_axes_locatable(axes[2, 3])
        cax0 = ax0_divider.append_axes("right", size="7%", pad=0.07)
        cbar1 = fig.colorbar(a3, ax=axes[2, 3], cax=cax0)

        # showing tumor core
        slice_tc_gt = imgs[7][0][:, :, slices]
        axes[3, 0].imshow(slice_tc_gt)
        axes[3, 0].axis('off')
        axes[3, 0].set_title(f'Tomur Core{title[0]}-slice{slices}', fontsize=22)
        # showing wohle tomur
        slice_tc_gt = imgs[7][0][:, :, slices]
        axes[3, 1].imshow(slice_tc_gt)
        axes[3, 1].axis('off')
        axes[3, 1].set_title(f'Whole Tomur{title[0]}-slice{slices}', fontsize=22)
        # showing wohle tomur pred
        slice_tc_prd = imgs[8][0][:, :, slices]
        axes[3, 2].imshow(slice_tc_prd)
        axes[3, 2].axis('off')
        axes[3, 2].set_title(f'prediction{title[0]}-slice{slices}', fontsize=22)
        # showing wt unc map
        slice_tc_unc = imgs[9][0][:, :, slices]
        a3 = axes[3, 3].imshow(slice_tc_unc, cmap=cm.coolwarm)
        axes[3, 3].axis('off')
        axes[3, 3].set_title(f'unc_map{title[0]}-slice{slices}', fontsize=22)
        ax0_divider = make_axes_locatable(axes[3, 3])
        cax0 = ax0_divider.append_axes("right", size="7%", pad=0.07)
        cbar1 = fig.colorbar(a3, ax=axes[3, 3], cax=cax0)

        # showing enhance tumor
        slice_et_gt = imgs[10][0][:, :, slices]
        axes[4, 0].imshow(slice_et_gt)
        axes[4, 0].axis('off')
        axes[4, 0].set_title(f'Enhance Tomur{title[0]}-slice{slices}', fontsize=22)
        # showing enhance tomur
        slice_et_gt = imgs[10][0][:, :, slices]
        axes[4, 1].imshow(slice_et_gt)
        axes[4, 1].axis('off')
        axes[4, 1].set_title(f'Enhance Tomur{title[0]}-slice{slices}', fontsize=22)
        # showing enhance tomur pred
        slice_et_prd = imgs[11][0][:, :, slices]
        axes[4, 2].imshow(slice_et_prd)
        axes[4, 2].axis('off')
        axes[4, 2].set_title(f'prediction{title[0]}-slice{slices}', fontsize=22)
        # showing et unc map
        slice_et_unc = imgs[12][0][:, :, slices]
        a3 = axes[4, 3].imshow(slice_et_unc, cmap=cm.coolwarm)
        axes[4, 3].axis('off')
        axes[4, 3].set_title(f'unc_map{title[0]}-slice{slices}', fontsize=22)
        ax0_divider = make_axes_locatable(axes[4, 3])
        cax0 = ax0_divider.append_axes("right", size="7%", pad=0.07)
        cbar1 = fig.colorbar(a3, ax=axes[4, 3], cax=cax0)

    if path:
        fig.savefig(path)

    # fig.tight_layout()
    plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()
    plt.close()


def print_unc(unc):
    max_unc_tot = np.max(unc, axis=(1, 2, 3))

    min_unc_tot = np.min(unc, axis=(1, 2, 3))

    mean_unc_tot = np.mean(unc, axis=(1, 2, 3))

    for i in range(len(max_unc_tot)):
        # print(f'img{i}, max_unc:{max_unc[i]:0.5f}')

        # print(f'img{i}, max_unc_b:{max_unc_b[i]:0.5f}')

        print(f'img{i}, min_unc_tot:{min_unc_tot[i]:0.5f}')

        print(f'img{i}, max_unc_tot:{max_unc_tot[i]:0.5f}')

        print(f'img{i}, mean_unc_tot:{mean_unc_tot[i]:0.5f}')


#####################
# save weights

def get_weight_samples(exp, opt, gpu, Nsamples=0):
    """
       path: path to weights that we have saved during training

       Nsamples: Number of samples that we want to use to plot a diagram

    """
    device = torch.device(gpu)
    weight_vec = []

    sve_dir = '/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/brats'

    weight_set_samples = torch.load(os.path.join(sve_dir + '/ckpts', f'{opt}_{exp}_state_dicts.pt'),
                                    map_location=device)

    if Nsamples == 0 or Nsamples > len(weight_set_samples):
        Nsamples = len(weight_set_samples)

    for idx, state_dict in enumerate(weight_set_samples):

        if idx == Nsamples:
            break

        for key in state_dict.keys():

            if 'weight' in key:

                weight_mtx = state_dict[key].cpu().data

                for weight in weight_mtx.view(-1):
                    weight_vec.append(weight)

    print(f'saving weights!')
    np.save(os.path.join(sve_dir + f'/weight_samples/{opt}_{exp}_{Nsamples}nsample_weights.npy'), weight_vec)


#####################


def normal_scaled(m, scale):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight, mean=0.0, std=1.0 * scale)

        torch.nn.init.normal_(m.bias, mean=0.0, std=1.0 * scale)

    #######################


def adjust_learning_rate(epoch, batch_idx):
    rcounter = epoch * num_batch + batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5 * cos_out * lr_0
    return (lr)


#######################
# labels in dataset

def brats_segmentation_regions():
    """
       based on challenge website 0 is for everything esle.
       https://www.med.upenn.edu/sbia/brats2018/tasks.html
    """

    dic_regions = {"NCR-NET": 1, "ED": 2, "ET": 3}
    return (dic_regions)


def _copy_input(input):
    if torch.is_tensor(input):
        return input.detach().clone()
    else:
        return input.copy()


# taking different regins
def get_wt(seg_map):
    """
       WT (yellow) : NCR-NET,ED,ET
    """

    seg_map_copy = _copy_input(seg_map)

    seg_map_copy[seg_map_copy != 0] = 1

    return (seg_map_copy.astype(np.int8))


def get_tc(seg_map):
    """
    TC (red): NCR-NET, ET

    """
    regions = brats_segmentation_regions()
    seg_map_copy = _copy_input(seg_map)

    # remove edema
    seg_map_copy[seg_map_copy == regions["ED"]] = 0

    seg_map_copy[seg_map_copy > 0] = 1

    return (seg_map_copy.astype(np.int8))


def get_et(seg_map):
    """
    ET (light-blue): ET

    """
    regions = brats_segmentation_regions()
    seg_map_copy = _copy_input(seg_map)

    seg_map_copy[seg_map_copy != regions["ET"]] = 0

    seg_map_copy[seg_map_copy > 0] = 1

    return (seg_map_copy.astype(np.int8))


def get_ed(seg_map):
    """
       ED: ED (yellow)
    """
    regions = brats_segmentation_regions()

    seg_map_copy = _copy_input(seg_map)

    seg_map_copy[seg_map_copy != regions["ED"]] = 0

    return (seg_map_copy.astype(np.int8))


def get_ncr_net(seg_map):
    """
       NCR-NET: green
    """

    regions = brats_segmentation_regions()

    seg_map_copy = _copy_input(seg_map)

    seg_map_copy[seg_map_copy != regions["NCR-NET"]] = 0

    return (seg_map_copy.astype(np.int8))


####################
# preprocess & augmentation
class Resize(object):

    def __init__(self, in_size):
        self.in_D, self.in_H, self.in_W = in_size

    def __call__(self, vol):
        if len(vol.shape) == 4:
            vol = vol.squeeze()

        [depth, height, width] = vol.shape
        scale = [self.in_D * 1.0 / depth, self.in_H * 1.0 / height, self.in_W * 1.0 / width]
        vol = ndimage.interpolation.zoom(vol, scale, order=0)

        return (vol)


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, vol):
        vol = (vol - vol.mean()) / (vol.std())
        return (vol)


def get_transform(in_size=64):
    in_size = in_size if isinstance(in_size, tuple) else (in_size, in_size, in_size)

    transform = transforms.Compose([Resize((in_size))])

    return (transform)


####################
# loading dataset

class BrainTum(Dataset):

    def __init__(self, root_dir, phase, transform=None, multimodal=False, corrupt=False, sigma=None, ood=False):

        self.root_dir = root_dir

        self.multimodal = multimodal

        self.corrupt = corrupt

        self.sigma = sigma

        self.ood = ood

        case_list = [os.path.join(self.root_dir, phase, case) for case in
                     os.listdir(os.path.join(self.root_dir, phase))]

        if phase == 'val':
            case_list = case_list[:-1]

        # case[0] gives us the Flair image
        # case[1] gives us seg mask

        self.flair_files = sorted(glob.glob(case + "/*_flair.nii.gz")[0] for case in case_list)
        self.seg_files = sorted(glob.glob(case + "/*_seg.nii.gz")[0] for case in case_list)

        if self.multimodal:
            self.t1_files = sorted(glob.glob(case + "/*_t1.nii.gz")[0] for case in case_list)
            self.t1ce_files = sorted(glob.glob(case + "/*_t1ce.nii.gz")[0] for case in case_list)
            self.t2_files = sorted(glob.glob(case + "/*_t2.nii.gz")[0] for case in case_list)

        self.transform = transform

        self.len = len(self.flair_files)

        assert len(self.flair_files) == len(self.seg_files)

    def __len__(self):

        return (self.len)

    def __getitem__(self, index):

        # the shape is in the form (depth,highet,width,channel)

        self.flair = nib.load(self.flair_files[index])
        self.seg = nib.load(self.seg_files[index])
        # converting to np array
        self.flair = np.array(self.flair.dataobj)
        self.seg = np.array(self.seg.dataobj)
        self.seg[self.seg == 4] = 3
        # print(f'number of classes in mask:{np.unique(self.seg)}')

        if self.multimodal:
            self.t1 = nib.load(self.t1_files[index])
            self.t1ce = nib.load(self.t1ce_files[index])
            self.t2 = nib.load(self.t2_files[index])
            # converting to np array
            self.t1 = np.array(self.t1.dataobj)
            self.t1ce = np.array(self.t1ce.dataobj)
            self.t2 = np.array(self.t2.dataobj)

            # self.ignore = self.mask>1
        # self.mask[self.ignore]=0

        if self.transform:

            self.flair = self.transform(self.flair)
            self.seg = self.transform(self.seg)

            if self.multimodal:
                self.t1 = self.transform(self.t1)
                self.t1ce = self.transform(self.t1ce)
                self.t2 = self.transform(self.t2)

        if self.corrupt:
            print(f'corrupt: {self.corrupt}')
            self.flair = gaussian_filter(self.flair, sigma=self.sigma)
            self.t1 = gaussian_filter(self.t1, sigma=self.sigma)
            self.t1ce = gaussian_filter(self.t1ce, sigma=self.sigma)
            self.t2 = gaussian_filter(self.t2, sigma=self.sigma)

        self.flair = Normalize()(self.flair)

        if self.multimodal:
            self.t1 = Normalize()(self.t1)
            self.t1ce = Normalize()(self.t1ce)
            self.t2 = Normalize()(self.t2)
            stacked_array = np.stack([self.t1ce, self.flair, self.t1, self.t2], axis=0)
            # Sara implimentation : np.stack((flair, t1, t2, t1_ce))

            if self.ood:
                print(f'ood: {self.ood}')
                stacked_array = np.stack([self.flair, self.t1, self.t2, self.t1ce], axis=0)

            self.vol = stacked_array
            self.seg = one_hot(self.seg, n_classes=4)

            # self.vol: [4,128,128,128]
            # self.seg: [128,128,128]: 0-4

        else:

            assert len(self.flair.shape) == 3
            assert len(self.seg.shape) == 3

            # self.vol: [1,128,128,128]
            # self.seg: [1,128,128,128]: 0-4
            self.vol = self.flair[np.newaxis, ...]
            self.seg = self.seg[np.newaxis, ...]

        ## Here I changed tensors to float32 because I got an error regarding double type
        self.vol = self.vol.astype('float32')
        self.seg = self.seg.astype('float32')

        return (self.vol, self.seg)


################################
class val_data(Dataset):

    def __init__(self, val_dir):
        self.val_dir = val_dir
        self.imglist = sorted(os.listdir(os.path.join(val_dir, 'imgs')))
        self.masklist = sorted(os.listdir(os.path.join(val_dir, 'masks')))

        self.len = len(self.imglist)

    def __len__(self):
        return (self.len)

    def __getitem__(self, index):
        img = torch.load(os.path.join(f'{self.val_dir}/imgs', self.imglist[index]))
        mask = torch.load(os.path.join(f'{self.val_dir}/masks', self.masklist[index]))

        return (img, mask)


#################################
# getting dataloader

## getting dataloader
def get_train_val_loader(dataset, batch_size, num_workers, val_size=0, val_dir=None):
    if val_size:

        assert val_size < 1., "Invalid argument for split: {}".format(val_split)

        ## calculate splits

        split = ShuffleSplit(n_splits=1, test_size=val_size, random_state=0)

        # Create lists
        indices = range(len(dataset))

        for train_index, val_index in split.split(indices):
            train_ind = train_index
            val_ind = val_index

        train_set = Subset(dataset, train_ind)

        val_set = Subset(dataset, val_ind)

        if val_dir:
            # save val_set
            img_path = val_dir + '/imgs'
            mask_path = val_dir + '/masks'
            os.makedirs(img_path, exist_ok=True)
            os.makedirs(mask_path, exist_ok=True)

            for i, (vol, mask) in enumerate(val_set):
                torch.save(vol, f'{img_path}/img{i}')
                torch.save(mask, f'{mask_path}/mask{i}')

        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                  shuffle=True)

        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)

        print(f'val_loader: {type(val_loader)}')

        return (train_loader, val_loader)

    else:

        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

        return (data_loader)


################################
# defining opimizer
## def optimizer

class SGLD(Optimizer):
    """
    SGLD based on pytorch's SGD
    Note that weight decay is specified based on the gaussian prior sigma
    Weight decay is L2 regularization
    """

    def __init__(self, params, lr=required, temp=1.0, weight_decay=0.0, addnoise=True, N_train=0):

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value:{}".format(weight_decay))

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid leraning rate:{}".format(lr))

        if temp < 0:
            raise ValueError('temp {%.3f} must be positive'.format(temp))

        if N_train <= 0:
            raise ValueError('You must provide total_sample_size to any SGD_MCMC method')

        defaults = dict(lr=lr, weight_decay=weight_decay, temp=temp, addnoise=addnoise, N_train=N_train)

        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):

        """a single optimization step"""

        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            temp = group['temp']
            N_train = group['N_train']

            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                if group['addnoise']:

                    noise = torch.randn_like(p.data).mul_((temp * group['lr'] / N_train) ** 0.5)
                    p.data.add_(d_p.data, alpha=-0.5 * group['lr'])

                    p.data.add_(noise)

                    if torch.isnan(p.data).any(): exit('Nan param')

                    if torch.isinf(p.data).any(): exit('inf param')

                else:

                    p.data.add_(0.5 * d_p, alpha=-group['lr'])
        return (loss)


###############################

DEFAULT_DAMPENING = 0.0


class SGHM(Optimizer):

    def __init__(self, params,
                 lr=required,
                 momentum=0.99,
                 dampening=0.,
                 weight_decay=0.,
                 N_train=0.,
                 temp=1.0,
                 addnoise=True):

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value:{}".format(weight_decay))

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid leraning rate:{}".format(lr))

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        N_train=N_train,
                        temp=temp,
                        addnoise=addnoise)

        super(SGHM, self).__init__(params, defaults)

    def step(self, closure=None):

        """a single optimization step"""

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            N_train = group['N_train']
            temp = group['temp']

            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad

                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                d_p.mul_(-(1 / 2) * group['lr'])

                if momentum != 0:

                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()

                    else:

                        buf = param_state['momentum_buffer']

                        buf.mul_(momentum * (group['lr'] / N_train) ** 0.5).add_(d_p, alpha=1 - dampening)

                    d_p = buf

                if group['addnoise']:

                    noise = torch.randn_like(p.data).mul_((temp * group['lr'] * (1 - momentum) / N_train) ** 0.5)

                    p.data.add_(d_p + noise)

                    if torch.isnan(p.data).any(): exit('Nan param')

                    if torch.isinf(p.data).any(): exit('inf param')

                else:

                    p.data.add_(d_p)
        return (loss)


##################################
class pSGLD(Optimizer):
    """
    pSGLD based on pytorch's SGD
    Note that weight decay is specified based on the gaussian prior sigma
    Weight decay is L2 regularization
    """

    def __init__(self, params, lr=required, weight_decay=0.,
                 alpha=0.99, eps=1e-7, centerd=False,
                 temp=1.0, addnoise=True, N_train=0., timestep_factor=1.0):

        # temp: float>0, temprature
        # timestep_factor: variable which can be used for learning rate decay.
        # A 'timestep_factor' of 0.5 would halve the SDE discretization time step

        """Raises:
                 ValueError: invalid argument value.
        """

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value:{}".format(weight_decay))

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid leraning rate:{}".format(lr))

        if temp < 0:
            raise ValueError('temp {%.3f} must be positive'.format(temp))

        if N_train <= 0:
            raise ValueError('You must provide N_train to any SGD_MCMC method')

        defaults = dict(lr=lr, weight_decay=weight_decay,
                        alpha=alpha, eps=eps, centerd=centerd,
                        addnoise=addnoise, temp=temp,
                        N_train=N_train, timestep_factor=timestep_factor)

        super(pSGLD, self).__init__(params, defaults)

    ## This exits in orginal implimentation of pytorch
    def __setstate__(self, state):

        super(pSGLD2CN, self).__setstate__(state)

        for group in self.param_groups:
            group.setdefault('centerd', False)

    def step(self, closure=None):

        """a single optimization step"""

        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            temp = group['temp']
            N_train = group['N_train']
            timestep_factor = group['timestep_factor']

            for p in group['params']:

                ### Note that p.grad gives us the grad for each layer weights and bias seperately
                ### for example if we have 2 conv and 1 dense layer in our network then
                ### p.grad: grad/conv1.weight
                ### p.grad: grad/conv1.bias
                ### p.grad: grad/conv2.weight
                ### p.grad: grad/conv2.bias
                ### p.grad: grad/fc.weight
                ### p.grad: grad/fc.bias
                ### so p.grad returns gradinate w.r.t each layer's weights and bias
                if p.grad is None:
                    continue

                # here we set p.grad instead of p.grad.data
                d_p = p.grad.data

                ## in the first epoch state is NULL:{}
                state = self.state[p]

                # initialize step
                # in the beginig of the program we should initialize v_p =0 for every p in
                # model.parameters
                # or equivalrently p in group['params'] for group in opt.param_groups

                if len(state) == 0:
                    state['step'] = 0

                    # note that state['square_avg'] is v_p =0 for every layer's parameter
                    # note that here it defines square_avg as a tensor with the same size
                    # and type as p
                    state['square_avg'] = torch.zeros_like(p.data)

                    if group['centerd']:
                        ## grad_avg is defined as a tensor as the same size and type of p
                        state['grad_avg'] = torch.zeros_like(p.data)

                ## if len(state)!= 0 menas that we are not at the begining of training
                ## so we value square_avg with current square_avg
                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                if weight_decay != 0:
                    # when we have weight decay, weight decay is multiplied by parameter
                    # and is added to the grdainat
                    # Note that torch.add(b,a,alpha = weight_decay), multiplies a, alpha
                    # and then add this multiplication to tensor b
                    # we can ake it in_place by writing b.add_(a, alhpa=weight_decay)

                    ## here in another implimentation it was p.data instead of p
                    d_p.add_(p.data, alpha=weight_decay)

                ## Updating running avg_sqr

                ## update squared_avg of gradient
                ## sqavg x alpha + (1-alph) p.grad*p.grad (elementwise)
                ## alpha * sqr_avg +(1-alpha)* p.grad**2
                ## square_avg.shape = d_p.shape = p.data.shape
                square_avg.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)

                if group['centerd']:
                    grad_avg = state['grad_avg']
                    # alpha*grad_avg+(1-alpha)* grad
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    # avg = square_avg-(grad_avg*gradavg(elenmentwise))=
                    # square_avg - grad_avg**2
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])

                else:

                    ## avg here is the same as sigma_s in the wenzel paper that is sigma_s = sqrt(v_s)+eps
                    avg = square_avg.sqrt().add_(group['eps'])

                # print(f'avg_shape:{avg.shape}')

                if group['addnoise']:

                    # torch.randn_like(p.data) : it returns a tensor with the same size as input
                    # with random numbers from normal distribution with mean:0 and variance:1
                    # note that mean is not so clsoe to 0, and varinace is not also very close to 1

                    # noise: noise = sqrt(T*lr/n).N(0,1)
                    # noise: noise = sqrt(T*lr*timestep_factor/N_train)
                    noise = torch.randn_like(p.data).mul_((temp * group['lr'] * timestep_factor / N_train) ** 0.5)

                    # updating noise as: sqrt(T * lr * timestep_factor/n). M^{-1/2}. N(0,1)
                    # noise_p = sqrt(T * lr* timestep_facor* M^{-1}). N(0,1)
                    noise /= torch.sqrt(avg)

                    ## update grad by multiplying to M^{-1}
                    ## d_p = d_p.M_p^{-1} = M^{-1} * grad
                    d_p.data.div_(avg)
                    ## d_p = -1/2 .lr. M^{-1}. d_p = -1/2 .lr. M^{-1}. grad
                    p.data.add_(d_p.data, alpha=-0.5 * group['lr'])

                    p.data.add_(noise)

                    if torch.isnan(p.data).any(): exit('Nan param')

                    if torch.isinf(p.data).any(): exit('inf param')

                    # langevin_noise =  p.data.new(p.data.size()).normal_(mean=0.0, std=1.0) / np.sqrt(group['lr'])

                    # Note that here when we wnt to add noise to our aparmeter, we add it to p.data
                    # because we do not want to track the gradinat
                    # or we can do this by:
                    # with torch.no_grad:
                    #     \theta += my_noise_tensor

                else:

                    # 0.5 *d_p is devided by avg elementwise, then is mutiplied by value
                    # and added to p

                    # p =p - 0.5* lr * grad_p * avg^{-1}
                    p.addcdiv_(0.5 * d_p, avg, value=-group['lr'])
        return (loss)


####################################
# defining unet model
# Defining the model with MonteCarloDropout

# # input: B,C,H,W,D
# output: B,C,H,W,D

class Conv(nn.Module):

    def __init__(self, in_channel, out_channel, drop=None, bn=True, padding=1, kernel=3, activation=True):

        super(Conv, self).__init__()

        self.conv = nn.Sequential()

        self.conv.add_module('conv', nn.Conv3d(in_channel, out_channel, kernel_size=kernel, padding=padding))

        if drop is not None:
            self.conv.add_module('dropout', nn.Dropout3d(p=drop, inplace=True))
        if bn:
            self.conv.add_module('bn', nn.BatchNorm3d(out_channel))

        if activation:
            self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return (x)


class DoubleConv(nn.Module):

    def __init__(self, in_channel, out_channel, mid_channel=None, drop=None, drop_mode='all', bn=True, repetitions=2):

        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel

        convs = []

        in_ch_temp = in_channel

        for i in range(repetitions):
            do = _get_dropout(drop, drop_mode, i, repetitions)

            convs.append(Conv(in_ch_temp, mid_channel, do, bn))

            in_ch_temp = mid_channel
            mid_channel = out_channel

        self.block = nn.Sequential(*convs)

    def forward(self, x):

        ## The spatial dim is not changed: (out_dim, h,w)
        return (self.block(x))


def _get_dropout(drop, drop_mode, i, repetitions):
    if drop_mode == 'all':
        return (drop)

    if drop_mode == 'first' and i == 0:
        return (drop)

    if drop_mode == 'last' and i == repetitions - 1:
        return (drop)

    if drop_mode == 'no':
        return (None)

    return (None)


def _get_dropout_mode(drop_center, curr_depth, depth, is_down):
    if drop_center is None:
        return 'all'

    if curr_depth == depth:
        return 'no'

    if curr_depth + drop_center >= depth:
        return 'last' if is_down else 'first'
    return 'no'


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channel, out_channel,
                 drop=None, drop_center='all', curr_depth=0, depth=4, bn=True):
        super().__init__()

        do_mode = _get_dropout_mode(drop_center, curr_depth, depth, True)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channel, out_channel, drop=drop, drop_mode=do_mode, bn=bn))

    def forward(self, x):
        ## here our spatial dimension would be divided by two
        return (self.maxpool_conv(x))


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channel, out_channel,
                 drop=None, drop_center='all', curr_depth=0, depth=4, bn=True, bilinear=True):

        super(Up, self).__init__()

        do_mode = _get_dropout_mode(drop_center, curr_depth, depth, False)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            ## if we ran into a problem we should use: drop = driop, drop_mode = do_mode, bn = bn
            ## because in DoubleConv we have some ambiguty due to repetisions =2
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2, drop, do_mode, bn)

        else:

            self.up = nn.ConvTranspose3d(in_channel, in_channel // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channel, out_channel, drop=drop, drop_mode=do_mode, bn=bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHWD
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        # print(f'diffY:{diffY}')
        # print(f'diffX:{diffX}')
        # print(f'diffZ:{diffZ}')

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        # print(f'pad: {[diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2, diffZ // 2,diffZ - diffZ // 2]}')

        x = torch.cat([x2, x1], dim=1)

        # print(f'x.size():{x.size()}')
        return (self.conv(x))


class OutConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return (self.conv(x))


class UNet2(nn.Module):
    DEFAULT_DEPTH = 4
    DEFAULT_DROPOUT = 0.2
    DEFAULT_FILTERS = 64

    def __init__(self, n_channels, n_classes, n_filters=DEFAULT_FILTERS, depth=DEFAULT_DEPTH, drop=DEFAULT_DROPOUT,
                 drop_center=None, bn=True, bilinear=False):
        super(UNet2, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.bilinear = bilinear
        self.drop = drop
        self.drop_center = drop_center
        self.bn = bn

        curr_depth = 0
        do_mode = _get_dropout_mode(drop_center, curr_depth, depth, True)
        self.inc = DoubleConv(n_channels, n_filters, drop=drop, drop_mode=do_mode, bn=bn)
        curr_depth += 1
        self.down1 = Down(n_filters, n_filters * 2, drop, drop_center, curr_depth, depth, bn)
        curr_depth += 1
        self.down2 = Down(n_filters * 2, n_filters * 4, drop, drop_center, curr_depth, depth, bn)
        curr_depth += 1
        self.down3 = Down(n_filters * 4, n_filters * 8, drop, drop_center, curr_depth, depth, bn)

        factor = 2 if bilinear else 1
        self.down4 = Down(n_filters * 8, n_filters * 16 // factor, drop, drop_center, depth, depth, bn)
        curr_depth = 3
        self.up1 = Up(n_filters * 16, n_filters * 8 // factor, drop, drop_center, curr_depth, depth, bn, bilinear)
        curr_depth = 2
        self.up2 = Up(n_filters * 8, n_filters * 4 // factor, drop, drop_center, curr_depth, depth, bn, bilinear)
        curr_depth = 1
        self.up3 = Up(n_filters * 4, n_filters * 2 // factor, drop, drop_center, curr_depth, depth, bn, bilinear)
        curr_depth = 0
        self.up4 = Up(n_filters * 2, n_filters, drop, drop_center, curr_depth, depth, bn, bilinear)

        ## Note that in the last conv of last layer we do not use any bn or dropout
        self.outc = OutConv(n_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

        # print(f'Downsampling 1!')
        # print(f'first d-conv:x1 {x1.size()}')
        x2 = self.down1(x1)

        # print(f'Downsampling 2!')
        # print(f'f_dwon+d-conv:x2 {x2.size()}')

        x3 = self.down2(x2)

        # print(f'S_dwon+d-conv:x3 {x3.size()}')
        # print(f'Downsampling 3!')

        x4 = self.down3(x3)
        # print(f'T_down+d-conv:x4 {x4.size()}')

        # print(f'Downsampling 4!')
        x5 = self.down4(x4)

        # print(f'Upsampling 1!')
        # print(f'F_down+d-conv:x5 {x5.size()}')
        x = self.up1(x5, x4)
        # print(f'Upsampling2!')
        # print(f'F_up: {x.size()}')
        x = self.up2(x, x3)
        # print(f'Upsampling3!')
        x = self.up3(x, x2)
        # print(f'Upsampling4!')
        x = self.up4(x, x1)
        logits = self.outc(x)

        return (logits)

    def sample_predict(self, x, Nsamples, classes=None):
        # print(f'Taking samples for mcd!')

        b, ch, h, w, d = x.size()

        predictions = x.data.new(Nsamples, b, ch, h, w, d)

        for i in range(Nsamples):
            # dim_y: [batch_size,n_calsses,length, width] = [16,32,64,64]
            y = self.forward(x.float())

            predictions[i] = y

            # if i> 0:

            # sanity chcek if we are taking different samples or not
            # print(f'sample {i} is eqaul to sample {i-1} :{torch.all(predictions[i].eq(predictions[i-1]))}')

        return (predictions)


#################################

## Dice loss
class DiceCoef(nn.Module):

    def __init__(self):

        super(DiceCoef, self).__init__()

        """
        Dice coef over the batch.
        Dividing by num_samples gives dice for each image
        """

    def forward(self, pred, target, smooth=1.):

        # applying softmax on logits to compute mean IOU
        # note that in the main version we gave this softmax function
        # since I got negative values for loss I disabled this softmax function
        # prob = F.softmax(pred,dim=1)

        # pred in thr_rem :torch.float64 (16, 128, 128, 128)
        # target in thr_rem : np.float64 (16,128,128,128)

        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)

        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)

        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        # this computes mean IOU over batch
        # it means that is is sum(IOU_img)/num samples in each batch
        score = (2. * intersection + smooth) / (A_sum + B_sum + smooth)

        # print(f'mean IOU over batch in DiceCoef:{score:0.4f}')
        return (score)


## note that in dice loss we should apply activation function
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

        """
        Dice coef over the batch.
        Dividing by num_samples gives dice for each image
        """

    def forward(self, pred, target, smooth=1.):
        # When we want to use it as a Loss function we need to apply an
        # activation function on logits
        # prob = F.softmax(pred,dim=1)

        # prob = F.sigmoid(pred)

        # prob = torch.sigmoid(pred)

        # have to use contiguous since they may from a torch.view op

        # if we do not have any activation function in our main program, we sould
        # apply a softmax function on logits

        prob = F.softmax(pred, dim=1)

        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)

        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        # this computes mean IOU over batch
        # it means that is is sum(IOU_img)/num samples in each batch
        score = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
        # print(f'mean IOU over batch in DiceLoss:{score:0.4f}')

        # it returns mean loss IOU  over batch
        loss = 1 - score

        return (loss)


#######################################

# lr0 : the initial learning rate

# it_idx : the index of batch in whole training process---> idx_batch

# n_batch: Number of batches

# cycle_batch_length -----> cycle_length * n_batch

# min_v : minimum value that we want to reach at the end of each cycle

# n_sam_per_cycle

min_v = 0


def update_lr(lr0, batch_idx, cycle_batch_length, n_sam_per_cycle, optimizer):
    is_end_of_cycle = False

    prop = batch_idx % cycle_batch_length

    pfriction = prop / cycle_batch_length

    print(f'pfriction: {pfriction}')

    lr = lr0 * (min_v + (1.0 - min_v) * 0.5 * (np.cos(np.pi * pfriction) + 1.0))

    if prop == 0:
        # sanity check of cyclic schedule learning rate
        print(f'Biggining of cycle : batch_idx: {batch_idx}, lr: {lr:0.7f}')

    if prop >= cycle_batch_length - n_sam_per_cycle:
        is_end_of_cycle = True

        print(f'Batch_sample: {batch_idx}, lr: {lr:0.7f}')

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return (is_end_of_cycle)


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return (group['lr'])


######################################
parser = argparse.ArgumentParser(description='TRAINING SG_MCMC FOR Brain Tumor Segmentation')

# argument experiment
parser.add_argument('--exp', type=int, default=49,
                    help='ID of this expriment!')

# parser.add_argument('--gpu',type=int,default=3,
#                        help='The id of free gpu for training.')

parser.add_argument('--n-epochs', type=int, default=2,
                    help='The number of epochs for training.')

parser.add_argument('--resume', type=bool, default=False,
                    help='If we want to resume a model or not.')

# arguments for leraning rate schedule
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Initial learning rate for training.')

parser.add_argument('--b-size', type=int, default=2,
                    help='The number of batch size for training.')

parser.add_argument('--n-workers', type=int, default=4,
                    help='Number of workers for loading data.')

# argument for loss
parser.add_argument('--crit', type=str, default='dice', choices=('Crsent', 'Focal', 'dice', 'BCrsent', 'comb'),
                    help='Loss function for training.')

# arguments for optimizer
parser.add_argument('--opt', type=str, default='adam',
                    choices=('sgld', 'sghm', 'psgld', 'psghm', 'sgd', 'adam', 'rmsprop'),
                    help='MCMC methods, one of: SGLD, SGHM, pSGLD, pSGHM')

parser.add_argument('--mom', type=float, default=0.99,
                    help='momentum decay for optimizer')

parser.add_argument('--arch', type=str, default='unet2', choices=('unet', 'unet2', 'unet3', 'deeplab'),
                    help='model for training.')

parser.add_argument('--prior', type=str, default=None, choices=(None, 'norm'),
                    help='if we wnat to use prior or not.')

# argumnets for model
parser.add_argument('--depth', type=int, default=4,
                    help='Depth of Unet model.')

parser.add_argument('--n-filter', type=int, default=64,
                    help='Depth of Unet model.')

parser.add_argument('--bi', type=bool, default=False,
                    help='Using upsampling in Unet model.')

parser.add_argument('--in-size', type=int, default=128,
                    help='Size of images for training.')

parser.add_argument('--plot', type=bool, default=True,
                    help='Plot learning curves.')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function for unet model.')

parser.add_argument('--temp', type=float, default=1.0,
                    help='Temperature used in MCMC scheme (used for sgld and sghm).')

parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight-decay.')

parser.add_argument('--add-noise', type=bool, default=False,
                    help='Adding noise to the optimizer or not')

parser.add_argument('--scale', type=bool, default=False,
                    help='If scale loss func or not')

parser.add_argument('--dr', type=float, default=None,
                    help='Dropout rate.')

# arguments for MCMC sampling
parser.add_argument('--sam-st', type=int, default=15,
                    help='Epoch that sampling is started.')

parser.add_argument('--lr-sch', type=str, default=None, choices=(None, 'fixed', 'cyclic'),
                    help='Type of learning rate schedule.')

parser.add_argument('--cycle-length', type=int, default=2,
                    help='cycle that we sample weights.')

parser.add_argument('--n-sam-cycle', type=int, default=1,
                    help='Number of samples that we wnat to take in ecah cycle!')

parser.add_argument('--n-ensemble', type=int, default=20,
                    help='The number of sample weights.')

# arguments for dataset
parser.add_argument('--dts', type=str, default='brats',
                    help='Dataset name.')

parser.add_argument('--save-sample', type=bool, default=False,
                    help='If we want to take samples or not.')

parser.add_argument('--multi-mod', type=bool, default=False,
                    help='If we want to work on a moltimodal or unimodal dataset.')

parser.add_argument('--pr', type=bool, default=False,
                    help='If we want to load pretrained model or not.')

####################################################################

CLIP_NORM = 0.25
DEFAULT_ALPHA = 0.99
DEFAULT_EPSILON = 1e-7
DEFAULT_DAMPENING = 0.0


def main(args):
    ### Here name is the name of experiment that I want to perform
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    exp = args.exp
    # gpu = args.gpu
    n_epochs = args.n_epochs
    lr0 = args.lr
    b_size = args.b_size
    n_workers = args.n_workers
    crit = args.crit
    opt = args.opt
    prior = args.prior
    mom = args.mom
    arch = args.arch
    n_filter = args.n_filter
    bi = args.bi
    in_size = args.in_size
    plot = args.plot
    activation = args.act
    temp = args.temp
    weight_decay = args.weight_decay
    addnoise = args.add_noise
    scale = args.scale
    dr = args.dr
    sampling_start = args.sam_st
    cycle_length = args.cycle_length
    N_samples = args.n_ensemble
    save_sample = args.save_sample
    dts = args.dts
    multimodal = args.multi_mod
    resume = args.resume
    lr_sch = args.lr_sch
    n_sam_cycle = args.n_sam_cycle
    print(f'resume:{resume}')

    print(f'bi :{bi}')

    print(f'if cuda is availble: {torch.cuda.is_available()}')
    # device=torch.device(gpu)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device)
    # logger.info(f"Device: {device}")

    transform = get_transform(in_size=in_size)

    if dts == 'brats':
        path_data = '/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Inputs/brats'
        save_dir = '/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/brats'
        res_dir = save_dir + '/results'

        # save_dir = '/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/brats/ckpts'
        dataset = BrainTum(path_data, phase='train', transform=transform, multimodal=multimodal)

    os.makedirs(save_dir, exist_ok=True)

    # save hyperparemtrs as json file
    param_dict = vars(args)

    # save and print hyperparametrs
    print(param_dict)
    with open(os.path.join(save_dir + '/params', f'{opt}_{exp}_params.json'), "w") as f:
        json.dump(param_dict, f, indent=4)

    #### DATASET
    # logger.info("Creating dataset")
    # save val set
    save_val = os.path.join(save_dir + '/val_loader', f'{opt}_{exp}')
    print(f'save_val:{save_val}')
    os.makedirs(save_val, exist_ok=True)

    train_loader, val_loader = get_train_val_loader(dataset, batch_size=b_size, num_workers=n_workers, val_size=0.1,
                                                    val_dir=save_val)

    # defining prior for weights
    # we scale std if we scale loss
    N_train = len(train_loader.dataset)

    print(f'length of train set: {N_train}')

    scale = N_train ** 0.5

    def normal(m):

        if type(m) == nn.Conv3d:
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0 * scale)

            torch.nn.init.normal_(m.bias, mean=0.0, std=1.0 * scale)

    # scaling weight_decay with number of samples if we have weight decay
    # weight_decay =1 is equal to N(mu=0, sigma=1)
    if weight_decay and scale:
        weight_decay = (weight_decay / N_train)
        print(f'weight decay is scaled with train set size!')

    img, mask = dataset[0]
    # logger.info(f"Image shape {img.shape}, Mask shape {mask.shape}")

    # logger.info(f'len trainset {train_loader.dataset}')
    # logger.info(f'len valset {val_loader.dataset}')

    ##### Model
    # logger.info("Initiating Model...")

    if arch == 'unet2':

        model = UNet2(n_channels=4, n_classes=4, n_filters=n_filter, drop=dr, bilinear=bi).to(device)
        print(f'Unet2 with drop {dr} was generated!')

        if prior == 'norm':
            assert scale, "If loss is not scaled, a scaled prior might be harmful"

            model.apply(normal)

            print(f'weights are initialized with scaled normal distribution!')

    ##### Optimizer
    # logger.info("Initializing optimizer")

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr0, weight_decay=weight_decay)

    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=mom, weight_decay=weight_decay)

    elif opt == 'rmsprop':

        optimizer = optim.RMSprop(model.parameters(), lr=lr0, weight_decay=weight_decay, momentum=mom)

    elif opt == 'sgld':

        optimizer = SGLD(params=model.parameters(),
                         lr=lr0,
                         temp=temp,
                         weight_decay=weight_decay,
                         addnoise=addnoise,
                         N_train=N_train)
    elif opt == 'sghm':
        optimizer = SGHM(params=model.parameters(),
                         lr=lr0,
                         temp=temp,
                         weight_decay=weight_decay,
                         addnoise=addnoise,
                         momentum=mom,
                         dampening=DEFAULT_DAMPENING,
                         N_train=N_train)

    elif opt == 'psgld':
        optimizer = pSGLD(params=model.parameters(),
                          lr=lr0,
                          alpha=DEFAULT_ALPHA,
                          eps=DEFAULT_EPSILON,
                          centerd=False,
                          temp=temp,
                          weight_decay=weight_decay,
                          addnoise=addnoise,
                          N_train=N_train,
                          timestep_factor=1)

    # logger.info(f'Optimzer {opt} with weight_decay {weight_decay} and temp {temp} and addnoise {addnoise} is generated')

    ##### Loss
    if crit == 'dice':
        loss = DiceLoss().to(device)

    elif crit == 'BCrsent':
        loss = nn.BCEWithLogitsLoss().to(device)

    elif crit == 'Crsent':
        loss = nn.CrossEntropyLoss().to(device)

    # logger.info(f"loss function {crit} is used for training")

    ##### performnace metric
    Dice = DiceCoef()

    #### lr schedule
    n_batch = len(train_loader)
    cycle_batch_length = cycle_length * n_batch
    batch_idx = 0
    print(f'cycle_batch_length: {cycle_batch_length}')

    #### resuming a saved model

    if resume:

        print(f'resume:{resume}')
        checkpoint_path = os.path.join(save_dir + '/ckpts', f'{opt}_{exp}')
        # logger.info("Loading model from checkpoint..")
        checkpoints = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoints['epoch']
        best_loss = checkpoints['val_loss']
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        print(f'model is loaded from epoch :{start_epoch}')

        weight_set_samples = torch.load(os.path.join(save_dir, f'/ckpts/{opt}_{exp}_state_dicts.pt'),
                                        map_location=device)
        sampled_epochs = torch.load(os.path.join(save_dir, f'/epochs/{opt}_{exp}_epochs.pt'), map_location=device)
        # logger.info(f'len samples saved from previous example is: {len(weight_set_samples)}')

    else:
        start_epoch = 0
        best_loss = 1000
        weight_set_samples = []
        sampled_epochs = []

    #### Train
    loss_total = {'train': [], 'val': []}
    dice_total = {'train': [], 'val': []}

    for epoch in range(start_epoch, n_epochs):

        print(f'Epoch:{epoch}: {get_lr(optimizer):0.7f}')

        # time for training and val
        tic = time.time()
        for phase in ['train', 'val']:

            # with torch.set_grad_enabled(phase == 'train'):
            if phase == 'train':
                model.train()
                dataloader = train_loader

            else:
                model.eval()
                dataloader = val_loader

            # sanity check to see if dropout is active durinf training
            # for m in model.modules():
            # if m.__class__.__name__.startswith('Dropout'):
            # print(f'Dropout during {phase} is : {m.training}')

            total_loss = 0
            total_dice = 0

            for j, (vol, mask) in enumerate(dataloader):

                # vol: torch.float32  [b_size, 4,64,64,64]
                # mask: torch.float32[b_size,4,64,64,64]
                vol = vol.to(device)
                mask = mask.to(device)
                gt = mask.argmax(1)

                # out: torch.float32  [b_size,4,64,64,46]
                # pred: torch.float32 [b_size,64,64,64]

                # print(f'model device: {next(model.parameters()).is_cuda}')

                out = model(vol)
                pred = out.argmax(1)

                # here I give target to dice score, since we have mask as [b_size,64,64,64]
                dice_t = Dice(gt, pred)
                total_dice += dice_t.item()

                # out: torch.float32    [b_size, 4,64,64,64]
                # target: torch.float32 [b_size,4, 64,64,64]

                if crit == 'Crsent':
                    target = gt.long()
                    loss_t = loss(out, target)

                elif crit == 'dice':
                    target = mask
                    loss_t = loss(out, target)

                elif crit == 'comb':
                    loss_t1 = nn.CrossEntropyLoss()(out, gt.long())
                    loss_t2 = DiceLoss()(out, mask)
                    loss_t = loss_t1 + loss_t2

                if scale and phase == 'train':
                    loss_t = loss_t * N_train

                    total_loss += loss_t.item() * vol.shape[0] / N_train

                elif scale and phase == 'val':

                    total_loss += loss_t.item() * vol.shape[0]

                else:

                    # we do not scale loss
                    total_loss += loss_t.item()

                if phase == 'train':

                    optimizer.zero_grad()

                    if lr_sch == 'cyclic':
                        is_end_of_cycle = update_lr(lr0, batch_idx, cycle_batch_length, n_sam_cycle, optimizer)

                    loss_t.backward()

                    # print(f'gradients are updated!')

                    # clipping loss before updating gradients
                    if lr_sch == 'cyclic':
                        ## clip gradinat by norm to avoide exploding gradiants
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM, norm_type=2)

                    optimizer.step()

                    # number of iterations
                    batch_idx += 1
                    # print(f'Epoch:{epoch}, Batch:{batch_idx}, {get_lr(optimizer):0.7f}')

            if scale:
                # devidng loss
                loss_total[phase].append(total_loss / len(dataloader.dataset))
            else:
                # computing loss and dice
                loss_total[phase].append(total_loss / len(dataloader))

            dice_total[phase].append(total_dice / len(dataloader))

            if save_sample:

                if lr_sch == 'cyclic':

                    if epoch >= sampling_start and is_end_of_cycle and phase == 'train':

                        if len(weight_set_samples) >= N_samples:
                            weight_set_samples.pop(0)
                            sampled_epochs.pop(0)

                        weight_set_samples.append(copy.deepcopy(model.state_dict()))
                        sampled_epochs.append(epoch)
                else:

                    if epoch >= sampling_start and epoch % cycle_length == 0 and phase == 'train':

                        if len(weight_set_samples) >= N_samples:
                            weight_set_samples.pop(0)
                            sampled_epochs.pop(0)

                        weight_set_samples.append(copy.deepcopy(model.state_dict()))
                        sampled_epochs.append(epoch)
                        # logger.info(f'sample {len(weight_set_samples)} from {N_samples} was taken!')

        # end of one epoch
        toc = time.time()
        runtime_epoch = toc - tic

        print('Epoch:%d, loss_train:%0.4f, loss_val:%0.4f, dice_train:%0.4f, dice_val:%0.4f, time:%0.4f seconds' % \
              (epoch, loss_total['train'][epoch], loss_total['val'][epoch], \
               dice_total['train'][epoch], dice_total['val'][epoch], runtime_epoch))

        # saving chcekpoint
        is_best = bool(loss_total['val'][epoch] < best_loss)
        best_loss = loss_total['val'][epoch] if is_best else best_loss
        checkpoints = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_loss,
            'val_dice_score': dice_total['val'][epoch]}

        torch.save(checkpoints, os.path.join(save_dir + '/ckpts', f'{opt}_{exp}'))
        # logger.info(f'Best model in epoch {epoch} was saved!')

    # logger.info(f'{len(weight_set_samples)} samples were taken!')
    state = pd.DataFrame({'train_loss': loss_total['train'], 'valid_loss': loss_total['val'],
                          'train_dice': dice_total['train'], 'valid_dice': dice_total['val']})

    state.to_csv(res_dir + f'/Loss/{opt}_{exp}_loss.csv')

    # save model at the end of training
    torch.save({'epoch': epoch,
                'lr': lr0,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'Loss': state}, os.path.join(save_dir + '/ckpts', f'{opt}_{exp}_seg3d.pt'))

    if save_sample:
        torch.save(weight_set_samples, os.path.join(save_dir + '/ckpts', f'{opt}_{exp}_state_dicts.pt'))
        torch.save(sampled_epochs, os.path.join(save_dir + '/epochs', f'{opt}_{exp}_epochs.pt'))
        print(f'sampled_epochs: {sampled_epochs}')

    if plot:
        # stats,titl,results_dir=None
        plotCurves(state, 'crit', os.path.join(res_dir + '/curves', f'{opt}_{exp}'))

    print(f'model is saved in:')
    print(os.path.join(save_dir + '/ckpts', f'{opt}_{exp}_seg3d.pt'))
    print(f'finish training')


###############################

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


##############################
## segmentation metrics

# Sensitivity: recall
def recall(tp, fn):
    """TP / (TP + FN)"""

    actual_positives = tp + fn

    if actual_positives <= 0:
        return (0)

    return (tp / actual_positives)


# Specificity: precision
def precision(tp, fp):
    """TP/ (TP + FP)"""
    predicted_positives = tp + fp

    if predicted_positives <= 0:
        return (0)

    return (tp / predicted_positives)


def fscore(tp, fp, tn, fn, beta: int = 1):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""
    assert beta > 0

    precision_ = precision(tn, fp)
    recall_ = recall(tp, fn)

    if ((beta * beta * precision_) + recall_) <= 0:
        return (0)

    fscore = (1 + beta * beta) * precision_ * recall_ / ((beta * beta * precision_) + recall_)
    return (fscore)


def accuracy(tp, fp, tn, fn):
    """(TP + TN) / (TP + FP + FN + TN)"""
    if (tp + fp + tn + fn) <= 0:
        return (0)
    return ((tp + tn) / (tp + fp + tn + fn))


def FDR(tp, fp):
    """ FP/ TP+FP"""
    predicted_positives = tp + fp

    if predicted_positives <= 0:
        return (0)

    return (fp / predicted_positives)


#############################
## Uncertainty metrics


def Entropy(p):
    """
       computing entropy for multimodal classification

       inputs: softmax probablities of form (tensor): [b_size (dts_size),n_classes,h,w,d]

       outputs: numpy_array float32: [b_size (dts_size),h,w,d]
    """
    cls = p.shape[1]

    H = -(p * np.log(p)).sum(axis=1)

    H_nr = H / np.log(cls)

    meanH = H.mean(axis=0)

    stdH = H.std(axis=0)

    return (H, H_nr)


def Entropy_b(p):
    """
       computing entropy for binary classification

       inputs: softmax probablities of form (tensor): [b_size(dts_size),n_classes,h,w,d]

       outputs: numpy_array float32: [b_size (dts_size),h,w,d]
    """
    p = p.cpu().numpy()

    H = -(p * np.log(p)).sum(axis=1)

    Hf = H

    p_b = (1 - p)

    # we need this line of code to prevent nan values
    p_b = np.where(p_b == 0, 0.0001, p_b)

    Hb = -(p_b * np.log(p_b)).sum(axis=1)

    Ht = Hf + Hb

    H_nr = Ht / np.log(2)

    meanH = Ht.mean(axis=0)

    stdH = Ht.std(axis=0)

    return (Hf, Hb, Ht, H_nr)


def Binary_Entropy(p):
    """
       computing entropy for binary classification

       inputs: softmax probablities of form (tensor): [b_size(dts_size),n_classes,h,w,d]

       outputs: numpy_array float32: [b_size (dts_size),h,w,d]
    """

    if not isinstance(p, np.ndarray):
        print(f'p is not a numpy array')
        p = p.cpu().numpy()

    p_f = p

    p_b = 1 - p

    p_b = np.where(p_b == 0, 1e-8, p_b)

    H = -(p_f * np.log(p_f) + p_b * np.log(p_b))

    # I normalize entropies but I do not return them
    H_nr = H / np.log(2)

    return (H)


def Variance_unc(prob_ens, unc_mod=None):
    """
       prob_enc: [n_ensemble,len(testset),cls,h,w,d]

       unc_mod: bool, True, if we want to compute unc for modalities

       output: np.array[b_size,h,w,d]
    """

    # computing unc on tomure regions

    wt_prob = np.sum(prob_ens[:, :, 1:], axis=2)
    tc_prob = np.sum(prob_ens[:, :, [1, 3]], axis=2)
    et_prob = prob_ens[:, :, 3]

    # computing var on tumore regions
    var_wt = np.var(wt_prob, axis=0)

    # computing unc on tc
    var_tc = np.var(tc_prob, axis=0)

    # computing unc on en
    var_et = np.var(et_prob, axis=0)

    if unc_mod == 'ent':
        ent_wt = Binary_Entropy(np.mean(wt_prob, axis=0))
        ent_tc = Binary_Entropy(np.mean(tc_prob, axis=0))
        ent_et = Binary_Entropy(np.mean(et_prob, axis=0))

        return (ent_wt, ent_tc, ent_et)

    return (var_wt, var_tc, var_et)


#########################
def fp_fn(preds, gts, uncs_thr=None, unc=False):
    """
       preds:   [dts_size, h,w,d]

       gts  :   [dts_size, h,w,d]

       uncs_thr: unc_map that was thersholded at thershold thr

    """

    tps = np.logical_and(gts, preds)
    tns = np.logical_and(~gts, ~preds)
    fps = np.logical_and(~gts, preds)
    fns = np.logical_and(gts, ~preds)

    tp = tps.sum()
    tn = tns.sum()
    fp = fps.sum()
    fn = fns.sum()

    if unc:

        tpu = np.logical_and(tps, uncs_thr)
        tnu = np.logical_and(tns, uncs_thr)
        fpu = np.logical_and(fps, uncs_thr)
        fnu = np.logical_and(fns, uncs_thr)

        tpu_s = tpu.sum()
        tnu_s = tnu.sum()
        fpu_s = fpu.sum()
        fnu_s = fnu.sum()
        return (tpu, tnu, fpu, fnu, tpu_s, tnu_s, fpu_s, fnu_s, tp, tn, fp, fn)

    else:

        return (tp, tn, fp, fn)


########################
## removing pixels based on uncertainties
def remove_thr(preds, uncs, gts, thersholds):
    """
    preds : predicted binary masks for subregions: (np.float64) [len(testset),h,w,d]

    uncs :  np.float64: [len(testset),h,w,d]

    gts  :  binary gts for subregions

    thersholds:  a list of float numbers as thersholds that could be one number or more than one number

    return:
           masked_imgs_dict: {thr: np.float()}, where np.float() is a numpy array of thersholded outputs
    """

    df = pd.DataFrame(index=thersholds, columns=['dice_mask', 'dice_rm', 'dice_add', 'dice_fp', 'dice_fn', 'dice_fp_fn',
                                                 'ref_vox_fp_fn', 'ret_vox_fp_fn', 'ftp', 'ftn', 'tpu_s',
                                                 'tnu_s', 'prd_mean',
                                                 'mask_prd_mean', 'tpu_fpu_ratio', 'tnu_fnu_ratio'])
    for thr in thersholds:
        print(f'thr:{thr}:')

        uncs = uncs.copy()
        preds_b = preds.copy().astype(np.bool)
        gts_b = gts.copy().astype(np.bool)

        # this is the mask that I want to use
        thersholded_unc = (uncs >= thr)

        tpu, tnu, fpu, fnu, tpu_s, tnu_s, fpu_s, fnu_s, tp, tn, fp, fn = fp_fn(preds_b, gts_b, thersholded_unc,
                                                                               unc=True)

        # computing dice for masked images

        # sanity check if masking works correctly or not
        preds_mean = preds.copy().mean()
        # masked_preds = np.ma.masked_array(preds.copy(), mask=thersholded_unc).data.astype(np.int)
        masked_preds = preds.copy()
        masked_gts = gts.copy()
        masked_preds[thersholded_unc] = 0
        masked_gts[thersholded_unc] = 0
        masked_preds_mean = masked_preds.mean()
        masked_dice = DiceCoef()(masked_preds, masked_gts)

        df.loc[thr]['dice_mask'] = masked_dice.item()
        df.loc[thr]['prd_mean'] = preds_mean.item()
        df.loc[thr]['mask_prd_mean'] = masked_preds_mean.item()

        # if tpu/fpu <1 we expect that dice becomes better
        # so I should look at this ratio to see that if dice helps or not
        # the samller ratio shows better metric closer to zero rather than 1
        tpu_fpu_ratio = tpu_s / fpu_s
        jaccard_index = tp / (tp + fp + fn)
        dice_benefit = (tpu_fpu_ratio < jaccard_index)

        # correct to background :
        corrected_preds = preds.copy()
        corrected_preds[thersholded_unc] = 0
        corrected_dice = DiceCoef()(corrected_preds, gts)
        print(dice_benefit)

        df.loc[thr]['dice_rm'] = corrected_dice.item()
        df.loc[thr]['tpu_fpu_ratio'] = tpu_fpu_ratio.item()

        # filtered true positive: (tp-tpu)/tp = 1-(tpu/tp) (1-auc2): ftp vs thr, auc2: ftp vs thr
        # filtered true negative: (tn-tnu)/tn = 1-(tnu/tn) (1-auc3): ftn vs thr, auc3: ftn vs thr
        ftp = (tp - tpu_s) / tp
        ftn = (tn - tnu_s) / tn

        df.loc[thr]['ftp'] = ftp
        df.loc[thr]['ftn'] = ftn
        df.loc[thr]['tpu_s'] = tpu_s
        df.loc[thr]['tnu_s'] = tnu_s

        # ratio of voxels that we thershold
        df.loc[thr]['ref_vox_fp_fn'] = (tpu_s + tnu_s + fpu_s + fnu_s) / (tp + tn + fp + fn)

        # ratio of voxels that retain
        df.loc[thr]['ret_vox_fp_fn'] = 1 - df.loc[thr]['ref_vox_fp_fn']

        # if tnu_s < fnu_s, then we should hopful in benefiting the dice score
        # so we should look at tnu_fnu_ratio ifit was samller than 1, then we should be hopefull for omrovement
        # the samller ratio shows better metric closer to zero
        tnu_fnu_ratio = tnu_s / fnu_s

        # correct to foreground :
        corrected_preds = preds.copy()
        corrected_preds[thersholded_unc] = 1
        corrected_add_dice = DiceCoef()(corrected_preds, gts)
        # print(f'dice_for : {corrected_add_dice.item()}')
        df.loc[thr]['dice_add'] = corrected_add_dice.item()

        df.loc[thr]['tnu_fnu_ratio'] = tnu_fnu_ratio.item()

        # correct fp to tn
        corrected_preds = preds.copy()
        corrected_preds[fpu] = 0
        corrected_dice_fp = DiceCoef()(corrected_preds, gts)
        # print(f'dice_fp : {corrected_dice_fp}')
        df.loc[thr]['dice_fp'] = corrected_dice_fp.item()

        # correct fn to tp
        corrected_preds = preds.copy()
        corrected_preds[fnu] = 1
        corrected_dice_fn = DiceCoef()(corrected_preds, gts)
        # print(f'dice_fn: {corrected_dice_fn}')
        df.loc[thr]['dice_fn'] = corrected_dice_fn.item()

        # correct (fp to tn) and (fn to tp)
        corrected_preds = preds.copy()
        corrected_preds[fpu] = 0
        corrected_preds[fnu] = 1
        corrected_dice_fp_fn = DiceCoef()(corrected_preds, gts)
        # print(f'dice_fp_fn: {corrected_dice_fp_fn}')

        df.loc[thr]['dice_fp_fn'] = corrected_dice_fp_fn.item()

    return (df)


def auc_score(df):
    df_score = pd.DataFrame(columns=['s_rm', 's_add', 's_mask', 's2', 's3', 'score_rm', 'score_add', 'score_msk'],
                            index=['score'])

    thersholds = df.index

    title = df_score.columns[5:]

    # auc dice vs thr
    s_rm = auc(thersholds, df['dice_rm'])

    s_add = auc(thersholds, df['dice_add'])

    s_mask = auc(thersholds, df['dice_mask'])

    dice_score = [s_rm, s_add, s_mask]

    df_score.loc['score']['s_rm'] = s_rm
    df_score.loc['score']['s_add'] = s_add
    df_score.loc['score']['s_mask'] = s_mask

    # auc ftp vs thr
    s2 = auc(thersholds, df['ftp'])
    df_score.loc['score']['s2'] = s2

    # auc ftn vs thr
    s3 = auc(thersholds, df['ftn'])
    df_score.loc['score']['s3'] = s3

    for s, tit in zip(dice_score, title):
        score = (s + (1 - s2) + (1 - s3)) / 3

        df_score.loc['score'][tit] = score

    return (df_score)


#########################

def roc_unc(probs, gts, uncs, thersholds):
    """ fdr = fp/ actuall(pos) = fp/ fp+tp"""

    b_thrs = [l.round(2) for l in list(np.arange(0.0, 1.1, 0.1))]

    df_tpr = pd.DataFrame(columns=b_thrs, index=thersholds)
    df_fdr = pd.DataFrame(columns=b_thrs, index=thersholds)

    for u_thr in thersholds:

        uncs_c = uncs.copy()

        print(f'u_thr: {u_thr}: {uncs_c.mean()}')

        mask = (uncs_c >= u_thr)

        print(f'mask:{mask.mean()}')

        for b_thr in b_thrs:
            print(f'##############')
            print(f'b_thr:{b_thr}')
            print(f'##############')

            pred_temp = np.zeros(probs.shape)

            probs_c = probs.copy()

            pred_temp = np.where(probs_c >= b_thr, 1.0, 0.0)

            pred_thr = pred_temp.copy()
            gts_thr = gts.copy()

            print(f'pred_thr:{pred_thr.mean()}')
            print(f'gts_thr: {gts_thr.mean()}')

            pred_thr[mask] = 0
            gts_thr[mask] = 0

            print(f'After masking:')
            print(f'pred_thr: {pred_thr.mean()}')
            print(f'gts_thr: {gts_thr.mean()}')

            pred_thr_b = pred_thr.astype(np.bool)
            gts_thr_b = gts_thr.astype(np.bool)

            tp, tn, fp, fn = fp_fn(pred_thr_b, gts_thr_b)

            print(f'tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}')

            tpr = recall(tp, fn)

            fdr = FDR(tp, fp)

            print(f'tpr:{tpr}')

            print(f'fdr:{fdr}')

            df_tpr.loc[u_thr][b_thr] = tpr

            df_fdr.loc[u_thr][b_thr] = fdr

    return (df_tpr, df_fdr)


##########################

def Unc_Thr(net, val_loader, Nsamples, unc, thr_mood, opt=None):
    """
    This function returns un_thr using validation set
    inputs
    :net: we need to evaluate the model on validation set
    :Nsamples: numebr of ensembels that we want to have
    :unc: 'MI', 'entropy', which uncertainty we want to use
    :thr_mood:, 'max': max and min uncertainity in valset is computed
                'mean': mean uncertainity on valset is computed
    """
    # set model in val mood
    net.set_model_train(False)

    if opt == 'mcd':
        net.model.apply(enable_dropout)

        print(f'Dropout is enabled!')

    unc_mean = 0

    unc_mean_img = np.zeros(len(val_loader.dataset))

    n_samples = 0

    with torch.no_grad():

        for j, (x, y, _) in enumerate(val_loader):

            print(f'batch {j} in val_loader!')

            prob_mean, prob_tot = net.all_sample_eval(x, y, Nsamples)
            entropy = Entropy(prob_mean.cpu().numpy())

            mi = MI(entropy, prob_tot.cpu().numpy())

            if unc == 'ent':
                uncertainty = entropy

            elif unc == 'mui':
                uncertainty = mi

            # max and min uncertainty on val set
            if j == 0:
                # initilizing unc_min, unc_max
                unc_min = uncertainty.min(axis=(0, 1, 2))
                unc_max = uncertainty.max(axis=(0, 1, 2))

            if j > 0:
                # updating the min, max values of uncertainty for each batch
                u_min = uncertainty.min(axis=(0, 1, 2))
                u_max = uncertainty.max(axis=(0, 1, 2))

                if u_min < unc_min:
                    unc_min = u_min
                if u_max > unc_max:
                    unc_max = u_max

            # mean all over pixels and samples in each batch
            unc_mean += uncertainty.mean(axis=(0, 1, 2))

            unc_mean_img[n_samples:n_samples + len(x)] = uncertainty.mean(axis=(1, 2))

            n_samples += len(x)

            # print uncertainty for image j
            print(f'min uncertainty for batch {j} is : {unc_min}')

            print(f'max uncertainty for batch {j} is : {unc_max}')

            # print(f'uncertainty for img {j} is :{uncertainty}')

            print(f'mean uncertainty for batch {j} is : {uncertainty.mean(axis=(0, 1, 2))}')

        un_thr_mean = unc_mean / len(val_loader)

        un_thr_mean2 = unc_mean_img.mean()

        print(f'un_thr_mean:{un_thr_mean:0.6f}')

        print(f'un_thr_mean2:{un_thr_mean2:0.6f}')

    if thr_mood == 'mean':
        un_thr = [un_thr_mean]

    elif thr_mood == 'max':
        un_thr = [unc_min, unc_max]

    return (un_thr)


########################

def true_positive(Acc, Unc):
    """
    it returns accurate and uncertian patches  Acc = [1,0,1,0,1]
                                               Unc = [0,1,1,0,0]
    :param Acc: list of acc [1: accurate, 0:inaccurate]
    :param Unc: list of unc [1: unceratin,0:certain]
    :return: number of true positive (acc_unc)
    """
    tp = 0

    for acc, unc in zip(Acc, Unc):

        if acc == 1 and unc == 1:
            tp += 1

    return (tp)


def true_negative(Acc, Unc):
    """
    it returns inaccurate and certain patches
    :param Acc: list of acc [1: accurate, 0:inaccurate]
    :param Unc: list of unc [1: unceratin,0:certain]
    :return: number of true negative (inac_c)
    """
    tn = 0

    for acc, unc in zip(Acc, Unc):

        if acc == 0 and unc == 0:
            tn += 1

    return (tn)


def false_positive(Acc, Unc):
    """
    it returns inaccurate and uncertain patches
    :param Acc: list of acc [1: accurate, 0:inaccurate]
    :param Unc: list of unc [1: unceratin,0:certain]
    :return: number of false positive (inac_c)
    """
    fp = 0

    for acc, unc in zip(Acc, Unc):

        if acc == 0 and unc == 1:
            fp += 1

    return (fp)


def false_negative(Acc, Unc):
    """
    it returns accurate and certain patches
    :param Acc: list of acc [1: accurate, 0:inaccurate]
    :param Unc: list of unc [1: unceratin,0:certain]
    :return: number of false negative (acc,c)
    """
    fn = 0

    for acc, unc in zip(Acc, Unc):

        if acc == 1 and unc == 0:
            fn += 1

    return (fn)


###########################################################

def extractPatches(unc_acc_map, window_shape=(4, 4, 4), stride=4):
    """input: im : np.float32: [h,w,d]
              window_shape:    [H,W,D]

       output:patches: np.float32:[nWindow,H,W,D]
    """

    assert len(window_shape) == 3, "shape of winow should be triple!"
    h, w, d = unc_acc_map.shape

    assert not (h % window_shape[0]) and not (w % window_shape[1]) and not (
                d % window_shape[2]), "The shape of image should be divisable by the shape of window!"

    patches = view_as_windows(unc_acc_map, window_shape, step=stride)

    nR, nC, nD, H, W, D = patches.shape

    nWindow = nR * nC * nD

    ## return np.float32 of shape: [n_window, h, w,ch]
    patches = np.reshape(patches, (nWindow, H, W, D))

    return (patches)


###########################################################

def Acc_bin_Unc_bin(Acc_arr, Unc_arr, acc_thr, un_thr):
    """
    This function lables each patch as un/certain, acc/inaccurate based on given thersholds
    inputs:
           Unc_arr: np.float: [n_patches], an array of float numbers(0<..<1) having the mean uncertainty of each patch

           Acc_arr: np.float: [n_patches], an array of float numbers (0<..<1) having the mean accuracy of ecah patch

           un_thr : a thershold for unceratinty labeling

           acc_thr: a thershold for accurate/inaccurate labeling
    return:
           Unc_bin: torch.float:[n_patches], a 0-1 array based on given thershold un_thr
           Acc_bin: torch.float:[n_patches], a 0-1 array based on given thershold acc_thr
    """

    Acc_bin = np.ones(Acc_arr.shape)

    Acc_bin[Acc_arr < acc_thr] = 0

    Unc_bin = np.ones(Unc_arr.shape)

    Unc_bin[Unc_arr < un_thr] = 0
    # print(f'max unc : {Unc_bin.max()}, min unc: {Unc_bin.min()}, unc_thr: {un_thr}')
    # print(f'Unc_bin: {Unc_bin}')
    # print(f'max acc : {Acc_bin.max()}, min acc: {Acc_bin.mean()}')

    # print(f'Acc_bin:{Acc_bin}')

    return (Acc_bin, Unc_bin)


##########################################################

def tp_tn_fp_fn(Acc_bin, Unc_bin):
    """
       inputs: Acc_bin: np.float64
               Unc_bin: np.float64

       conf_mat in sklearn is of the form:[tn=ic,fp=iu]
                                          [fn=ac,tp=au]

    """

    conf_mat = confusion_matrix(Acc_bin, Unc_bin)
    # print(f'conf_mat:\n{conf_mat}')

    # print(f'{conf_mat.ravel()}')
    # tn,fp,fn,tp
    ic, iu, ac, au = conf_mat.ravel()

    tp = true_positive(Acc_bin, Unc_bin)
    fp = false_positive(Acc_bin, Unc_bin)
    fn = false_negative(Acc_bin, Unc_bin)
    tn = true_negative(Acc_bin, Unc_bin)

    # print(f'au:{au}, tp:{tp}')
    # print(f'iu:{iu}, fp:{fp}')
    # print(f'ac:{ac}, fn:{fn}')
    # print(f'ic:{ic}, tn:{tn}')
    assert au == tp, "true positive dose not match"
    assert iu == fp, "false positive (inaccurate_uncertain) dose not match"
    assert ac == fn, "false negative (accurate and ceratin) dose not match"
    assert ic == tn, "true negative (inaccurate and certain) dose not match"

    return (au, iu, ac, ic)


################################################

def probabs(*a):
    """
    This function compute p(acc|certain), p(uncertain|inacc), p(PacvsPun)
    :params:
            a: is a numpy array : [au_t,iu_t,ac_t,ic_t,p1,p2,p3]
            au_t: float, number of accurate and uncertain patches
            iu_t: float, number of inaccurate and uncertain patches
            ac_t: float, number of accurate and certain patches
            ic_t: float, number of inaccurate and certain patches
    return:
           p(acc|certain), p(uncertain|inacc), p(PacvsPun)
    """
    au_t, iu_t, ac_t, ic_t = a[0][0:4]

    p_acc_con = ac_t / (ac_t + ic_t)

    p_unc_ina = iu_t / (iu_t + ic_t)

    p_ac_vs_un = (ac_t + iu_t) / (ac_t + iu_t + ic_t + au_t)

    return (p_acc_con, p_unc_ina, p_ac_vs_un)


###########################################################

def PAvsPU(Unc_t, Acc_t, acc_thr, thersholds, thr_mood, patch_dim=(4, 4, 4)):
    """
    inputs:
              Unc_t    : np.float32:[n_samples,h,w,d]

              Acc_t      :np.bool:   [n_samples,h,w,d]

              acc_thr  :thershold that we want to use for lableing each patch as accurate or inaccurate

             thersholds:list, of length>0. length =1, if thr_mood=mean
                                           length =2, if thr_mood=max
                                           length >0, if thr_mood=interval
             thr_mood  : 'max', min and max  uncertainity on valset
                         'mean',mean uncertainity on valset
                         'interval',a list of float numbers >0
      outputs:
              n_ac: number of accurate and ceratin patches
              n_au: number of accurate and uncertain patches
              n_ic: number of inaccurate and uncertain patches
              n_iu: number of inaccurate and uncertain patches

              p_acc_con: p(acc|conf)

              p_unc_ina: p(uncer|inacc)

              p_ac_vs_un: PAcvPU
    """

    n_samples = len(Unc_t)

    # print(f'n_samples:{n_samples}')

    print(f'thr_mood:{thr_mood}')

    print(f'patches of dim =({patch_dim}) are generated')

    if thr_mood == 'max':

        t = np.random.random(1)

        print(f't {t} for setting a thershold')
        print(f'un_min:{thersholds[0]:0.4f}, un_max:{thersholds[1]:0.4f}')
        # un_thr= un_min+(t*(un_max-un_min))
        un_thr = thersholds[0] + (t * (thersholds[1] - thersholds[0]))

    else:

        un_thr = thersholds
        if thr_mood == 'mean':
            print(f'un_mean:{un_thr[0]:0.3f}')

    # we round unc_thrs up to three decimal numbers
    un_thr = list(map(lambda a: round(a, 3), un_thr))
    print(f'un_thr:{un_thr}')

    L_t = np.zeros((len(un_thr), 7))

    for j, (Unc, Acc) in enumerate(zip(Unc_t, Acc_t)):
        # print(f'j:{j}')

        # Unc: np.float64 [h,w] = [128,128]
        # Acc: np.float32 [h,w] = [128,128]

        # Unc_patches :np.float [1024, 4,4,4]  here nWindows: 1024, each patch is of dimension: [4,4,4]
        # acc_patches :np.float [1024, 4,4,4]  here nWindows: 1024, each patch is of dimension: [4,4,4]
        Unc_patches = extractPatches(Unc, patch_dim, stride=patch_dim[0])

        Acc_patches = extractPatches(Acc, patch_dim, stride=patch_dim[0])

        max_unc = np.max(Unc_patches, axis=(1, 2, 3))
        min_unc = np.min(Unc_patches, axis=(1, 2, 3))

        # print(f'max_unc_bf_mean:{np.max(max_unc):0.4f}')

        # print(f'min_unc_bf_mean:{np.min(min_unc):0.4f}')

        # compute mean of uncertainty of each patch
        # Unc_patches_mean : [1024,]
        Unc_patches_mean = Unc_patches.mean(axis=(1, 2, 3))
        # compute mean of acc of each patch
        # Acc_patches_mean: [1024,]
        Acc_patches_mean = Acc_patches.mean(axis=(1, 2, 3))

        for i in range(len(un_thr)):
            # here we binerize Acc, Unc Maps

            Acc_bin, Unc_bin = Acc_bin_Unc_bin(Acc_patches_mean, Unc_patches_mean, acc_thr, un_thr[i])

            au, iu, ac, ic = tp_tn_fp_fn(Acc_bin, Unc_bin)

            L_t[i][0] += au
            L_t[i][1] += iu
            L_t[i][2] += ac
            L_t[i][3] += ic

    print(f'P(accurate|certain)=ac/(ac+ic)')
    print(f'P(uncertain|inaccurate)=iu/(iu+ic)')
    print(f'P(PAvPU)= ac+iu/(ac+ic+iu+au)')

    # the less value for the last metric shows better performance
    # the higher value for this metric is better
    print(f'ratio of accurate filtered patches = ac+au-au/ac+au=ac/ac+au')

    far = np.zeros(range(len(un_thr)))

    for i in range(len(un_thr)):
        p_acc_con, p_unc_ina, p_ac_vs_un = probabs(L_t[i])

        L_t[i][4] = p_acc_con

        L_t[i][5] = p_unc_ina

        L_t[i][6] = p_ac_vs_un

        # print(f'au: {L_t[i][0]}')
        # print(f'ac: {L_t[i][2]}')
        # far[i] = L_t[i][2]/(L_t[i][0]+L_t[i][2])

    df = pd.DataFrame(L_t, columns=['au', 'iu', 'ac', 'ic', 'P(acc|cer)', 'P(uncer|inacc)', 'PAvsPU'],
                      index=np.array(un_thr))
    df.index.name = 'un_thr'

    return (df)


#####################################################

def ECE(conf, acc, n_bins=5):
    """
    acc_bm = sigms 1(\hat{y}_i==y_i)/ |b_m|
    conf_bm= sigma \hat{pi} / |b_m|

    acc_bm == conf_bm


    """

    print(f'conf: {conf.shape}')
    print(f'acc:{acc.shape}')

    acc_list = []
    conf_list = []

    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    bin_lowers = bin_boundaries[:-1]

    bin_uppers = bin_boundaries[1:]

    ece = 0.0

    bin_counter = 0

    avg_confs_in_bins = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

        in_bin = np.logical_and(conf > bin_lower, conf <= bin_upper)

        # |B_m | /n
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:

            # acc_in_bin : |acc_m|/|B_m|

            acc_in_bin = np.mean(acc[in_bin])

            # avg_conf_in_bin: |conf_m|/|B_m|

            avg_conf_in_bin = np.mean(conf[in_bin])

            delta = avg_conf_in_bin - acc_in_bin

            avg_confs_in_bins.append(delta)

            acc_list.append(acc_in_bin)

            conf_list.append(avg_conf_in_bin)

            # |acc_m - conf_m|* |B_m|/n
            ece += np.abs(delta) * prop_in_bin

        else:

            avg_confs_in_bins.append(None)

        bin_counter += 1

        # For reliability diagrams, also need to return these:
        # return ece, bin_lowers, avg_confs_in_bins

    return (ece, acc_list, conf_list)


########################
# dummy functions

def epoch_index(epochs=300, Nsamples=80, cycle_len=3, sampling_srt=15):
    L = np.arange(epochs)

    idx = np.arange(Nsamples)
    epoch_indices = []

    for i in L:

        if i >= sampling_srt and i % cycle_len == 0:

            if len(epoch_indices) >= Nsamples:
                epoch_indices.pop()

            epoch_indices.append(i)

    dic = dict(zip(idx, epoch_indices))

    return (dic)


def sel_sampl(L, gap=2, cycle_len=3):
    cr_cycle = gap * cycle_len

    sel_epochs = {}

    for (key, l) in L.items():

        if l % cr_cycle == 0:
            sel_epochs[key] = l

    print("################")
    print(len(sel_epochs))
    print("################")

    print(f'{len(sel_epochs)} samples with cycle_length {cr_cycle} starting from 15 is generated!')

    return (sel_epochs)


##########################

## evaluation
def enable_dropout(m):
    if type(m) == nn.Dropout3d:
        m.train()


parser2 = argparse.ArgumentParser(description='TESTING SG_MCMC FOR BrainTomurSegmentation')

# argument experiment
parser2.add_argument('--exp', type=int, default=49,
                     help='ID of this expriment!')

parser2.add_argument('--gpu', type=int, default=3,
                     help='The id of free gpu for training.')

# arguments for optimizer
parser2.add_argument('--opt', type=str, default='adam',
                     choices=('sgld', 'sghm', 'psgld', 'psghm', 'sgd', 'adam', 'rmsprop'),
                     help='MCMC methods, one of: SGLD, SGHM, pSGLD, pSGHM')

parser2.add_argument('--sampler', type=str, default=None, choices=(None, 'sgmcmc', 'mcd', 'sgd'),
                     help='Sampler to pick MC sampels.')

parser2.add_argument('--n-ensemble', type=int, default=20,
                     help='The number of sample weights.')

# argumnets for architecture
parser2.add_argument('--arch', type=str, default='unet2', choices=('unet', 'unet2', 'unet3', 'deeplab'),
                     help='model for testing.')

parser2.add_argument('--dr', type=float, default=None,
                     help='Dropout rate.')

parser2.add_argument('--n-filter', type=int, default=64,
                     help='Depth of Unet model.')

parser2.add_argument('--act', type=str, default='relu',
                     help='activation function for unet model.')

parser2.add_argument('--bi', type=bool, default=False,
                     help='Using upsampling in Unet model.')

parser2.add_argument('--cls', type=int, default=4,
                     help='Number of modalities that we wnat to use.')

# argument for loss
parser2.add_argument('--crit', type=str, default='dice', choices=('Crsent', 'Focal', 'dice', 'BCrsent', 'comb'),
                     help='Loss function for test.')

# arguments for dataset
parser2.add_argument('--dts', type=str, default='brats',
                     help='Dataset name.')

parser2.add_argument('--in-size', type=int, default=128,
                     help='Size of images for testing.')

parser2.add_argument('--b-size', type=int, default=2,
                     help='The number of batch size for testing.')

parser2.add_argument('--multimod', type=bool, default=True,
                     help='If we use multimodal or unimodal!')

# arguments for metrics
parser2.add_argument('--metrics', type=str, default=None, choices=(
None, 'ece', 'pavpu', 'cavcu', 'rm_unc', 'unc_err', 'pavpu_m', 'ece_m', 'ece_s', 'conf_count', 'roc_unc'), nargs='*',
                     help='The metric for evaluating uncertainties.')

parser2.add_argument('--rm-thr', type=float, nargs='*',
                     help='Thresholds to thershold uncertainties!')

parser2.add_argument('--thr-m', type=str, default='interval', choices=('mean', 'max', 'interval'),
                     help='Thershold mood for pavpu.')

parser2.add_argument('--acc-thr', type=float, default=0.5,
                     help='Thershold for accuracy.')

parser2.add_argument('--vis', type=bool, default=True,
                     help='If we wnat to visulize the images and their uncertainty masp or not!')

parser2.add_argument('--vis2', type=int, nargs='*',
                     help='Number of images that we want to visualize!')

parser2.add_argument('--window', type=int, default=(2, 2, 2),
                     help='the size of extracted pathes for pavpu')

parser2.add_argument('--write-exp', type=bool, default=True,
                     help='If we write experiments in a csv file or not!')

parser2.add_argument('--unc', type=bool, default=True,
                     help='If we wnat to compute uncertainties or not!')

parser2.add_argument('--unc-mod', type=str, default=None, choices=(None, 'var', 'ent'),
                     help='If we wnat to compute uncertainties for modalities or not!')

parser2.add_argument('--thr-unc-err', type=float, nargs='*',
                     help='Thersholds that we set to check the overlap between seg error and nc_thr!')

parser2.add_argument('--thr-roc-unc', type=float, nargs='*',
                     help='Thersholds that we set to check if we can improve auc!')

parser2.add_argument('--dts-shift', type=str, default=None, choices=(None, 'corr', 'ood'),
                     help='If we want to shift dataset or not!')

parser2.add_argument('--sig-noise', type=int, default=None,
                     help='Noise that we want to inject to images!')

# arguments for selecting epochs
parser2.add_argument('--n-epochs', type=int, default=None,
                     help='Number of epochs for training!')

parser2.add_argument('--sam-st', type=int, default=15,
                     help='Epoch that sampling is started.')

parser2.add_argument('--cycle-length', type=int, default=3,
                     help='cycle that we sample weights.')

parser2.add_argument('--sel-epochs', type=bool, default=False,
                     help='if we want to take different epochs for ensembeling.')

parser2.add_argument('--gap', type=int, default=1,
                     help='the distance that we want to have between different samples!')


def test(args):
    seed = 42
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # random.seed(seed)
    # np.random.seed(seed)

    exp = args.exp
    gpu = args.gpu
    b_size = args.b_size
    crit = args.crit
    opt = args.opt
    arch = args.arch
    n_filter = args.n_filter
    bi = args.bi
    activation = args.act
    cls = args.cls
    in_size = args.in_size
    dr = args.dr
    Nsamples = args.n_ensemble
    dts = args.dts
    multimodal = args.multimod
    sampler = args.sampler
    vis = args.vis
    window = args.window
    metrics = args.metrics
    thr_m = args.thr_m
    rm_thr = args.rm_thr
    acc_thr = args.acc_thr
    write_exp = args.write_exp
    vis2 = args.vis2
    unc_mod = args.unc_mod
    unc = args.unc
    thr_unc_err = args.thr_unc_err
    dts_shift = args.dts_shift
    sig_noise = args.sig_noise
    n_epochs = args.n_epochs
    sam_st = args.sam_st
    cycle_length = args.cycle_length
    sel_epochs = args.sel_epochs
    gap = args.gap
    thr_roc_unc = args.thr_roc_unc

    # loading test set
    transform = get_transform(in_size=in_size)

    # set loading and saving path
    path_data = '/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/Inputs/brats'
    svd = '/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/brats'

    # save and print hyperparametrs
    param_dict = vars(args)
    print(param_dict)
    with open(os.path.join(svd + '/paramst', f'{opt}_{exp}_params.json'), "w") as f:
        json.dump(param_dict, f, indent=4)

    if dts_shift == 'corr':

        testset = BrainTum(path_data, 'val', transform, multimodal, corrupt=True, sigma=sig_noise)
        print("\n############################################################")
        print(f'Corrupted images by noise with sigma: {sig_noise} are loaded!')
        print("\n############################################################")

    elif dts_shift == 'ood':

        testset = BrainTum(path_data, 'val', transform, multimodal, ood=True)
        print("\n############################################################")
        print(f'OOD images are loaded!')
        print("\n############################################################")


    else:

        print(f'Normal test set was generated!')

        testset = BrainTum(path_data, 'val', transform, multimodal)

    test_loader = get_train_val_loader(testset, batch_size=b_size, num_workers=4, val_size=0, val_dir=None)

    print(f'len testset:{len(testset)}')
    print(f'number of batches: {len(test_loader)}')

    load_dir = os.path.join(svd, f'ckpts/{opt}_{exp}_seg3d.pt')
    print(f'load_dir:{load_dir}')

    device = torch.device(gpu)
    torch.cuda.set_device(device)

    if arch == 'unet2':
        model = UNet2(n_channels=4, n_classes=cls, n_filters=n_filter, drop=dr, bilinear=bi)
        print(f'Unet2 with drop {dr} was generated!')

    checkpoint = torch.load(load_dir, map_location=device)
    epoch = checkpoint['epoch']

    # load weights for sgld and sghm samplesr
    if sampler == 'sgmcmc' or sampler == 'sgd':
        weight_set_samples = torch.load(os.path.join(svd, f'ckpts/{opt}_{exp}_state_dicts.pt'), map_location=device)
        print(f'{len(weight_set_samples)} weights are loaded!')

        epoch_svd = svd + f'/epochs/{opt}_{exp}_epochs.pt'

        if os.path.exists(epoch_svd):
            sampled_epochs = torch.load(epoch_svd, map_location=device)
            assert len(weight_set_samples) == len(sampled_epochs), print(
                'The length of sampled weights and sampled epochs are not equal')

        else:

            sampled_epochs = []

    # load model.state_dict for MC_D
    else:
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        print(f'checkpoints are loaded!')

    print(f'model is evaluaed for {Nsamples} samples!')

    print("\n#################################")
    print('Pretrained model is loaded from {%d}th Epoch ' % (epoch))
    print("\n#################################")

    # defining loss
    if crit == 'dice':
        loss = DiceLoss().to(device)

    elif crit == 'BCrsent':
        loss = nn.BCEWithLogitsLoss().to(device)

    elif crit == 'Crsent':
        loss = nn.CrossEntropyLoss().to(device)

    # defining dice coef to evaluate seg performance
    Dice = DiceCoef()

    loss_total = {'train': [], 'val': []}
    dice_total = {'train': [], 'val': []}

    tic = time.time()
    total_loss = 0
    total_dice = 0

    print(testset[0][0].shape)
    ch, h, w, d = testset[0][0].shape

    n_batch = len(test_loader)
    test_set_size = n_batch * b_size

    # ensembeling on whole weight samples or a subset of samples

    if sel_epochs:
        # ensembeling on a subset of sampled weights
        sampled_dic = epoch_index(n_epochs, Nsamples, cycle_length, sam_st)

        sub_epochs = sel_sampl(sampled_dic, gap=gap, cycle_len=cycle_length)

    # entropy_total =np.zeros((len(testset),h,w,d))
    ent_tot = np.zeros((len(testset), h, w, d))
    ent_nor = np.zeros((len(testset), h, w, d))
    b_ent_tot = np.zeros((len(testset), h, w, d))

    var_tot = np.zeros((len(testset), cls, h, w, d))

    preds_tot = np.zeros((test_set_size, h, w, d))

    out_tot = np.zeros((len(testset), cls, h, w, d))
    prob_tot = np.zeros((len(testset), cls, h, w, d))
    confs_tot = np.zeros((len(testset), h, w, d))
    acc_tot = np.zeros((len(testset), h, w, d))

    if unc_mod:
        wt_unc_t = np.zeros((len(testset), h, w, d))
        tc_unc_t = np.zeros((len(testset), h, w, d))
        et_unc_t = np.zeros((len(testset), h, w, d))

        wt_gt_t = np.zeros((len(testset), h, w, d))
        wt_prd_t = np.zeros((len(testset), h, w, d))

        tc_gt_t = np.zeros((len(testset), h, w, d))
        tc_prd_t = np.zeros((len(testset), h, w, d))

        et_gt_t = np.zeros((len(testset), h, w, d))
        et_prd_t = np.zeros((len(testset), h, w, d))

    gts = []
    imgs = []
    n_samples = 0

    THERSHOLD = 0.5

    with torch.no_grad():
        model.eval()

        for j, (vol, mask) in enumerate(test_loader):
            # vol: torch.float32 [b_size, 4,64,64,64]
            # mask: torch.float32[b_size,4,64,64,64]
            vol = vol.to(device)
            mask = mask.to(device)

            # gt: torch.float32 [b_size,64,64,64]
            gt = mask.argmax(1)

            gts.extend(gt.cpu().detach().numpy())
            imgs.extend(vol.cpu().detach().numpy())

            if sampler == 'sgmcmc' or sampler == 'sgd':

                if sel_epochs:

                    out_t = vol.data.new(len(sub_epochs), b_size, ch, h, w, d)

                    for idx, i in enumerate(sub_epochs.keys()):

                        if idx == Nsamples:
                            break

                        model.load_state_dict(weight_set_samples[i])

                        model.to(device)

                        out_t[idx] = model(vol.float())

                else:

                    out_t = vol.data.new(Nsamples, b_size, ch, h, w, d)

                    for idx, weight_dict in enumerate(weight_set_samples):

                        if idx == Nsamples:
                            break

                        # sample_epoch = sampled_epochs[idx]
                        # print(f'sample from epoch: {sample_epoch}')

                        model.load_state_dict(weight_dict)

                        model.to(device)

                        out_t[idx] = model(vol.float())

                mean_out = out_t.mean(dim=0, keepdim=False)

                # out: torch.float32 ([2, 1, 128, 128, 128])
                out = mean_out

            elif sampler == 'mcd':

                # we must enable dropout here for MC sampling

                model.apply(enable_dropout)

                # sanity chcek to see if dropout is on or not
                # for m in model.modules():

                # if m.__class__.__name__.startswith('Dropout'):

                # print(f'Dropout is on: {m.training}')

                # mean_out: [2, 1, 128, 128, 128]
                out_t = model.sample_predict(vol, Nsamples).to(device)

                mean_out = out_t.mean(dim=0, keepdim=False)

                out = mean_out

            else:

                # out: torch.float32[b_size,cls,h,w,d]
                out = model(vol)

                # adding extra dimension to the output for computing the varinace of subregions
                # out_t: [1,b_size,h,w,d]
                out_t = out[None, :]

            if multimodal:

                # out: [b_size,ch,h,w,d]
                probs = F.softmax(out, dim=1).data
                confs, preds = probs.max(dim=1, keepdim=False)
                accs = preds.eq(gt)
                dice_t = Dice(preds, gt)

                # computing total prob for all samples to compute varience: [N_samples,b_size,cls,h,w,d]
                prob_t = F.softmax(out_t, dim=2).data.cpu().numpy()

                # computing gt and out for subregions
                wt_gt = get_wt(gt.cpu().numpy())
                wt_prd = get_wt(preds.cpu().numpy())
                wt_gt_t[n_samples:n_samples + len(vol), :] = wt_gt
                wt_prd_t[n_samples:n_samples + len(vol), :] = wt_prd

                tc_gt = get_tc(gt.cpu().numpy())
                tc_prd = get_tc(preds.cpu().numpy())
                tc_gt_t[n_samples:n_samples + len(vol), :] = tc_gt
                tc_prd_t[n_samples:n_samples + len(vol), :] = tc_prd

                et_gt = get_et(gt.cpu().numpy())
                et_prd = get_et(preds.cpu().numpy())
                et_gt_t[n_samples:n_samples + len(vol), :] = et_gt
                et_prd_t[n_samples:n_samples + len(vol), :] = et_prd

            else:

                probs = F.sigmoid(out).squeeze().data
                preds = torch.round(probs)
                accs = preds.eq(gt)
                dice_t = Dice(preds, gt)

                # computing total prob for all samples to compute varience: [N_samples,b_size,cls,h,w,d]
                prob_t = F.sigmoid(out_t, dim=2).data.cpu().numpy()

            # computing loss and dice for segmentation performance
            if crit == 'Crsent':

                target = gt.long()

            else:
                target = mask

            # out: torch.float32    [b_size,cls,h,w,d]
            # target: torch.float32 [b_size,cls,h,w,d]
            loss_t = loss(out, target)
            total_loss += loss_t.item()

            total_dice += dice_t.item()

            # out_tot:  [len(testset),cls,h,w,d]
            # preds_tot:[len(testset),h,w,d]
            # prob_tot: [len(testset),cls,h,w,d]
            # acc_tot:  [len(testset),h,w,d]
            # confs_tot:[len(testset),h,w,d]
            out_tot[n_samples:n_samples + len(vol), :] = out.detach().cpu().numpy()
            preds_tot[n_samples:n_samples + len(vol), :] = preds.detach().cpu().numpy()
            prob_tot[n_samples:n_samples + len(vol), :] = probs.detach().cpu().numpy()
            acc_tot[n_samples:n_samples + len(vol), :] = accs.cpu().numpy()
            confs_tot[n_samples:n_samples + len(vol), :] = confs.cpu().numpy()

            if unc:

                if multimodal:
                    # computing uncertainty uisng entropy
                    entropy_t, entropy_nr = Entropy(probs.cpu().numpy())
                else:
                    entropy_t, entropy_nr = Binary_Entropy(probs.cpu().numpy())

                ent_tot[n_samples:n_samples + len(vol), :] = entropy_t
                ent_nor[n_samples:n_samples + len(vol), :] = entropy_nr

                if unc_mod:
                    wt_unc, tc_unc, et_unc = Variance_unc(prob_t, unc_mod)

                    wt_unc_t[n_samples:n_samples + len(vol), :] = wt_unc
                    tc_unc_t[n_samples:n_samples + len(vol), :] = tc_unc
                    et_unc_t[n_samples:n_samples + len(vol), :] = et_unc

            n_samples += len(vol)

        # sanity check for prob, acc, conf
        # print(f'prob2==prob4:{prob_tot[2].all()==prob_tot[4].all()}')
        # print(f'conf2==conf4:{confs_tot[2].all()==confs_tot[4].all()}')
        # print(f'accs2==accs4:{acc_tot[2].all()==acc_tot[4].all()}')

        # saving out_tot,prob_tot,acc_tot,confs_tot
        np.save(svd + f'/results/out_tot/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_out_tot.npy', out_tot)
        np.save(svd + f'/results/prob_tot/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_prob_tot.npy', prob_tot)
        np.save(svd + f'/results/conf_tot/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_conf_tot.npy', confs_tot)
        np.save(svd + f'/results/acc_tot/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_acc_tot.npy', acc_tot)

        # computing loss and dice
        total_loss /= len(test_loader)
        total_dice /= len(test_loader)

        print("\n#################################")
        print(f'loss_test:{total_loss:0.4f} dice_test:{total_dice:0.4f}')
        print("\n#################################")

        # looking at max, min, mean uncertainty and probablities

        gts = np.array(gts)
        print(f'acc_tot: {type(acc_tot)}, {acc_tot.shape},{acc_tot.dtype}')

        df_p = pd.DataFrame(index=np.arange(len(testset)), columns=['min', 'max', 'mean'])
        # looking at prob_tot
        for prob in prob_tot:
            df_p['min'] = prob.min()

            df_p['max'] = prob.max()

            df_p['mean'] = prob.mean()

        print(f'dataframe of probablities :{df_p}')
        df_p.to_csv(svd + f'/results/prob/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_prob.csv', index=True)

        if unc:
            print(f'unc is used!')
            unc_t = ent_nor
            # looking at entropy
            min_unc, max_unc, mean_unc = Unc_stats(unc_t)
            ent_tot_df = pd.DataFrame({'min unc': min_unc, 'max unc': max_unc, 'mean unc': mean_unc})
            print(ent_tot_df)
            ent_tot_df.to_csv(svd + f'/results/unc/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_unc.csv', index=True)

        # computing dice coefficient and other metrics for each sub-region
        if multimodal:

            gts_sub = [wt_gt_t, tc_gt_t, et_gt_t]
            prds_sub = [wt_prd_t, tc_prd_t, et_prd_t]
            title = ['wt', 'tc', 'et']

            # computing probablities for each sub region
            wt_prob_p = np.sum(prob_tot[:, 1:], axis=1)
            tc_prob_p = np.sum(prob_tot[:, [1, 3]], axis=1)
            et_prob_p = prob_tot[:, 3]
            prob_sub = [wt_prob_p, tc_prob_p, et_prob_p]

            # check the dimention of probablities
            print(f'wt prob: {wt_prob_p.shape}')
            print(f'tc prob: {tc_prob_p.shape}')
            print(f'et prob: {et_prob_p.shape}')

            # computing acc_map for each sub region
            acc_wt = (wt_prd_t == wt_gt_t)
            acc_tc = (tc_prd_t == tc_gt_t)
            acc_et = (et_prd_t == et_gt_t)
            acc_sub = [acc_wt, acc_tc, acc_et]

            # computing metrics for sub regions
            df_eval = pd.DataFrame(columns=["dice", "recall", "precision", "fscore", "accuracy"], index=title)

            for i, (gt_s, prd_s) in enumerate(zip(gts_sub, prds_sub)):
                dice_s = Dice(gt_s, prd_s).item()

                # making them boolean
                gts_b = gt_s.astype(np.bool)
                preds_b = prd_s.astype(np.bool)

                tp, tn, fp, fn = fp_fn(preds_b, gts_b)

                # sensitivity = tpr= tp/tp+fn
                recall_s = recall(tp, fn)
                # specificity = tp/tp+fp
                prec_s = precision(tp, fp)

                fscore_s = fscore(tp, fp, tn, fn)

                acc_s = accuracy(tp, fp, tn, fn)

                df_eval.loc[title[i]]["dice"] = dice_s
                df_eval.loc[title[i]]["recall"] = recall_s
                df_eval.loc[title[i]]["precision"] = prec_s
                df_eval.loc[title[i]]["fscore"] = fscore_s
                df_eval.loc[title[i]]["accuracy"] = acc_s

            print("\n#####evaluation metrics for sub regions#####")
            print(df_eval)
            print("\n############################################")
            print(f'whole Tomure Dice score:  {df_eval.loc["wt"]["dice"]:0.4f}')
            print(f'Tomure Core Dice score:   {df_eval.loc["tc"]["dice"]:0.4f}')
            print(f'Enhance Tomure Dice score {df_eval.loc["et"]["dice"]:0.4f}')
            print("\n############################################")

            df_eval.to_csv(svd + f'/results/seg_sub/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_eval.csv', index=True)

            if unc_mod:
                uncs_sub = [wt_unc_t, tc_unc_t, et_unc_t]
                uncs_sub_nr = uncs_sub / np.log(2)

                # looking at max, min, mean modality uncertainty
                min_unc_wt, max_unc_wt, mean_unc_wt = Unc_stats(wt_unc_t)
                wt_unc_tot_df = pd.DataFrame(
                    {'min unc wt': min_unc_wt, 'max unc wt': max_unc_wt, 'mean unc wt': mean_unc_wt})
                print(wt_unc_tot_df)

                min_unc_tc, max_unc_tc, mean_unc_tc = Unc_stats(tc_unc_t)
                tc_unc_tot_df = pd.DataFrame(
                    {'min unc tc': min_unc_tc, 'max unc tc': max_unc_tc, 'mean unc tc': mean_unc_tc})
                print(tc_unc_tot_df)

                min_unc_et, max_unc_et, mean_unc_et = Unc_stats(et_unc_t)
                et_unc_tot_df = pd.DataFrame(
                    {'min unc et': min_unc_et, 'max unc et': max_unc_et, 'mean unc et': mean_unc_et})
                print(et_unc_tot_df)

                # saving min, max, mean uncertainty
                wt_unc_tot_df.to_csv(svd + f'/results/unc/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_wtunc.csv', index=True)
                tc_unc_tot_df.to_csv(svd + f'/results/unc/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_tcunc.csv', index=True)
                et_unc_tot_df.to_csv(svd + f'/results/unc/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_etunc.csv', index=True)

        # looking at output's varinace
        # print(f'var_tot shape: {var_tot.shape}')
        # min_var, max_var, mean_var = Var_stats(var_tot)
        # var_tot_df = pd.DataFrame({'min var': min_var, 'max var': max_var, 'mean var': mean_var})
        # print(var_tot_df)
        # var_tot_df.to_csv(svd+f'/results/unc/{opt}_exp_{exp}_{Nsamples}ens_{dr}dr_var.csv',index=True)

        if metrics:

            if 'rm_unc' in metrics:

                print(f'removing uncs in predicted mask')

                rm_thr = [l.round(2) for l in list(np.arange(0.1, 0.8, 0.1))]

                print(rm_thr)

                for i in range(3):
                    print(f'statistics for sub_region[{i}]:')
                    df = remove_thr(prds_sub[i], uncs_sub[i], gts_sub[i], rm_thr)
                    # print(dice_thr_df)
                    df.to_csv(svd + f'/results/rm_unc/{opt}_exp_{exp}_{title[i]}_ens_{Nsamples}_dr_{dr}_rm_unc.csv',
                              index=True)

                    score = auc_score(df)

                    print(df)
                    print(score)
                    score.to_csv(svd + f'/results/rm_unc/{opt}_exp_{exp}_{title[i]}_ens_{Nsamples}_dr_{dr}_score.csv', \
                                 index=True)

            if 'roc_unc' in metrics:

                print(f'Computing tpr and fdr in thersholded uncs')

                for i in range(3):
                    print(f'++++++++++++++++++++++++')
                    print(f'computing roc_unc for {title[i]}')
                    print(f'++++++++++++++++++++++++')

                    df_tpr_i, df_fdr_i = roc_unc(prob_sub[i], gts_sub[i], uncs_sub_nr[i], thr_roc_unc)

                    df_tpr_i.to_csv(
                        svd + f'/results/roc_unc/{opt}_exp_{exp}_{title[i]}_ens_{Nsamples}_dr_{dr}_tpr_unc.csv', \
                        index=True)

                    df_fdr_i.to_csv(
                        svd + f'/results/roc_unc/{opt}_exp_{exp}_{title[i]}_ens_{Nsamples}_dr_{dr}_fdr_unc.csv', \
                        index=True)

            if 'pavpu' in metrics:

                unc_thr = list(np.arange(0.1, 1.0, 0.1))
                df = PAvsPU(Unc_t=unc_t, Acc_t=acc_tot, acc_thr=acc_thr, thersholds=unc_thr, \
                            thr_mood=thr_m, patch_dim=window)
                df.to_csv(svd + f'/results/pavspu/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_win_{window[0]}_pavpu.csv',
                          index=True)
                print(df)

                if 'pavpu_m' in metrics:

                    unc_thr_sub = list(np.arange(0.1, 0.7, 0.1))

                    for i in range(3):
                        df_sub = PAvsPU(Unc_t=uncs_sub[i], Acc_t=acc_sub[i], acc_thr=acc_thr, thersholds=unc_thr_sub, \
                                        thr_mood=thr_m, patch_dim=window)

                        df_sub.to_csv(svd + f'/results/pavspu/{opt}_exp_{exp}_win_{window[0]}_pavpu_{title[i]}.csv',
                                      index=True)

            if 'unc_err' in metrics:

                # computing unc_err for whole image
                thr_unc_err_t = [l.round(2) for l in list(np.arange(0.1, 1.0, 0.1))]
                print(thr_unc_err_t)

                unc_thr_list = [(unc_t >= thr) for thr in thr_unc_err_t]

                acc_tot_b = acc_tot.astype(np.bool)

                err_seg_tot = ~acc_tot_b

                Dice_t = [Dice(err_seg_tot, unc_thr).item() for unc_thr in unc_thr_list]

                unc_err = pd.Series(Dice_t, index=thr_unc_err_t)

                unc_err.to_csv(svd + f'/results/unc_err/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_unc_err_tot.csv',
                               index=True)

                # computing unc_err for subregions
                thr_unc_err = [l.round(2) for l in list(np.arange(0.1, 0.7, 0.1))]

                print(thr_unc_err)

                gts_sub_b = list(map(lambda l: l.astype(np.bool), gts_sub))

                Dice_sub = pd.DataFrame(columns=title, index=thr_unc_err)

                for i in range(3):

                    err_seg = np.logical_and(~gts_sub_b[i], prds_sub[i])

                    for thr in thr_unc_err:
                        unc_thr = (uncs_sub[i] >= thr)

                        Dice_err = Dice(err_seg, unc_thr)

                        Dice_sub.loc[thr][title[i]] = Dice_err.item()

                    print(Dice_sub)

                Dice_sub.to_csv(svd + f'/results/unc_err/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_unc_err.csv',
                                index=True)

            if 'ece' in metrics:

                # confs_tot : [Nsamples,h,w,d]
                # acc_tot  : [Nsamples,h,w,d] ----> bool

                ece, acc, conf = ECE(confs_tot, acc_tot)
                print(f'ece:{ece}')

                np.save(svd + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_acc.npy', acc)
                np.save(svd + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_conf.npy', conf)
                np.save(svd + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_ece.npy', ece)

                if 'ece_s' in metrics:

                    idx = [2, 4, 6]

                    com2 = confs_tot[2] == confs_tot[4]
                    acc2 = acc_tot[2] == acc_tot[4]

                    print(f'conf_2= conf_4: {com2.all()}')
                    print(f'acc_2= acc_4: {acc2.all()}')

                    for i in idx:
                        print(f'subject {i}')

                        ece_i, acc_i, conf_i = ECE(confs_tot[i], acc_tot[i])
                        print(f'ece subject {i}:{ece_i}')
                        print(f'acc subjcet {i}:{acc_i}')
                        print(f'conf subject{i}:{conf_i}')

                        np.save(svd + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_sub_{i}_acc.npy', acc_i)
                        np.save(svd + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_sub_{i}_conf.npy', conf_i)
                        np.save(svd + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_sub_{i}_ece.npy', ece_i)

                if 'ece_m' in metrics:

                    # define wt prob, tc prob, et prob (pos, neg cls)
                    wt_prob = np.zeros(wt_prob_p.shape)
                    tc_prob = np.zeros(tc_prob_p.shape)
                    et_prob = np.zeros(et_prob_p.shape)

                    # define prob of both pos and neg classes
                    wt_prob = np.where(wt_prob_p > 1 - wt_prob_p, wt_prob_p, 1 - wt_prob_p)
                    tc_prob = np.where(tc_prob_p > 1 - tc_prob_p, tc_prob_p, 1 - tc_prob_p)
                    et_prob = np.where(et_prob_p > 1 - et_prob_p, et_prob_p, 1 - et_prob_p)

                    probs_sub = [wt_prob, tc_prob, et_prob]

                    for i in range(3):
                        ece_s, acc_s, conf_s = ECE(probs_sub[i], acc_sub[i])

                        print(f'ece {i}:{ece_s}')
                        print(f'acc {i}:{acc_s}')
                        print(f'conf{i}:{conf_s}')

                        np.save(svd + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_reg_{i}_acc.npy', acc_s)
                        np.save(svd + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_reg_{i}_conf.npy', conf_s)
                        np.save(svd + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_reg_{i}_ece.npy', ece_s)

            if 'conf_count' in metrics:

                # thersholds that we want to exclude confidances
                thersholds = list(np.arange(-0.1, 1.1, 0.1))

                thersholds = [np.round(thr, decimals=2) for thr in thersholds]

                print(f'thersholds: {thersholds}')

                L = []

                cdf = pd.DataFrame(index=thersholds)

                accuracies = np.zeros(len(thersholds))

                cdf_L = [(unc_t <= thr).mean() for thr in thersholds]

                cdf = pd.Series(cdf_L, index=thersholds)

                print(f'cdf: {cdf}')

                # computing CDF for entropy
                # unc_t = unc_t.copy()

                # print(f'unc for thr {thr} is {unc_t.mean()}')

                # print(f'mean unc_thr for thr {thr}: {(unc_t <= thr).mean()}, max unc_thr is {(unc_t <= thr).max()}')

                # unc_t_thr = (unc_t <= thr).mean()

                # cdf.loc[thr]= unc_t_thr

                for i, thr in enumerate(thersholds):
                    # number of pixels grater than each thershold

                    mask_tot = (confs_tot >= thr)

                    count = mask_tot.sum()

                    L.append(count)

                    accuracies[i] = np.ma.masked_array(acc_tot, mask=~mask_tot).mean()

                    unc_thr = (unc_t <= thr)

                    print(f'thr: {unc_thr.sum()}')

                    n_sam, h, w, d = unc_t.shape

                    print(f'number of all pixels: {n_sam * h * w * d}')

                    print(f'thershold: {thr} min unc_thr: {unc_thr.min()}, max unc_thr: {unc_thr.max()}')

                df_count = pd.DataFrame(L, index=thersholds)
                df_count.name = 'thershold'

                df_acc = pd.DataFrame(accuracies, index=thersholds)
                df_acc.name = 'thershold'

                print(f'count vs ther')

                print(df_count)

                print(f'acc vs ther')
                print(df_acc)

                print(f'cdf: {cdf}')

                df_count.to_csv(svd + f'/results/corr/{opt}_exp_{exp}_sig_{sig_noise}_conf.csv', index=True)

                df_acc.to_csv(svd + f'/results/corr/{opt}_exp_{exp}_sig_{sig_noise}_acc.csv', index=True)

                cdf.to_csv(svd + f'/results/corr/{opt}_exp_{exp}_sig_{sig_noise}_cdf.csv', index=True)

        if unc:

            imgs_list1 = [gts, preds_tot, ent_tot, ent_tot]
            title = ['ground_truth', 'preds', 'ent_map', 'ent_map']

        else:

            imgs_list1 = [gts, preds_tot]
            title = ['ground_truth', 'preds']

        if vis2:
            img = []
            mask = []
            pred = []
            ent = []

            for ind in vis2:
                img.append(imgs[ind])
                mask.append(gts[ind])
                pred.append(preds_tot[ind])
                ent.append(ent_tot[ind])

            if unc_mod:

                for ind in vis2:
                    img.append(imgs[ind])
                    mask.append(wt_gt_t[ind])
                    mask.append(tc_gt_t[ind])
                    mask.append(et_gt_t[ind])

                    img.append(imgs[ind])
                    pred.append(wt_prd_t[ind])
                    pred.append(tc_prd_t[ind])
                    pred.append(et_prd_t[ind])

                    img.append(imgs[ind])
                    ent.append(wt_unc_t[ind])
                    ent.append(tc_unc_t[ind])
                    ent.append(et_unc_t[ind])

            imgs_list2 = [img, mask, pred, ent]
            title = vis2

        # if vis:

        #    imgs, title = 'Orginal', cols = 6, rows = 1, plot_size=(16,16),slices=None
        #    visualize(
        #            svd+f'/results/preds/{opt}_exp_{exp}_{Nsamples}ens_pred.png',
        #    imgs=imgs,
        #    title =title,
        #    cols = 6,
        #    rows = 1,
        #    plot_size=(18,18),
        #    norm =False,
        #    slices=100,
        #    dts=dts)

        if vis2:
            rows = 5 if unc_mod else 2

            print(f'unc_mod: {unc_mod}')

            visualize3(
                svd + f'/results/preds/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_unc_{unc_mod}_dtshf_{dts_shift}_pred.png',
                imgs=imgs_list2,
                title=title,
                cols=4,
                rows=rows,
                plot_size=(20, 10),
                norm=False,
                slices=90,
                dts=dts)

        if write_exp:
            data = {
                'exp': [exp],
                'in_size': [in_size],
                'opt': [opt],
                'n_ensembel': [Nsamples],
                'NLL': [round(total_loss, 4)],
                'dice': [round(total_dice, 4)]}

        # ./Baysian_Seg/Results/
        csv_path = svd + f'/results/Loss/run_sweeps_test.csv'

        if os.path.exists(csv_path):

            sweeps_df = pd.read_csv(csv_path)
            sweeps_df = sweeps_df.append(
                pd.DataFrame.from_dict(data), ignore_index=True).set_index('exp')

        else:
            sweeps_df = pd.DataFrame.from_dict(data).set_index('exp')

        # save experiment metadata csv file
        sweeps_df.to_csv(csv_path)

# if __name__=='__main__':

#    args2 = parser2.parse_args()

#    test(args2)            
     






