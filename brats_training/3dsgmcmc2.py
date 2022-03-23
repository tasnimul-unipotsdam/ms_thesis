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

from torch import nn
import copy
import pandas as pd
from matplotlib import cm
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset
import nibabel as nib
from scipy.ndimage import label

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

from skimage.util import view_as_windows
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, \
    precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve

import time

import datetime
import torch.utils.data
from torchvision import transforms, datasets
import matplotlib as mlp
import matplotlib.pyplot as plt
import torch.optim as optim

torch.manual_seed(42)
torch.cuda.manual_seed(42)

try:
    import cPickle as pickle
except:
    import pickle

import warnings

warnings.filterwarnings('ignore')


## commands to manage the memory


## Making a sample to test the functions

def Sample():
    PATH = '/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/Inputs/panceras/train'
    img_list = [img for img in os.listdir(PATH)]

    mask_list = [mask for mask in os.listdir(f'{PATH}' + "_labels")]

    sample = np.load(os.path.join(PATH, img_list[0]))

    mask = np.load(os.path.join(f'{PATH}' + "_labels", mask_list[0]))

    print(sample.shape)
    print(mask.shape)

    return (sample, mask)


####################
def visualize2(path=None,
               imgs=None,
               title=None,
               cols=4,
               rows=2,
               plot_size=(8, 8),
               norm=False,
               slices=50,
               dts=None,
               tr='inter'):
    fig, axes = plt.subplots(rows, cols, figsize=plot_size)

    if tr == 'padd':

        d1 = 8
        d2 = 8 + 112

        h1 = 43
        h2 = 43 + 42

        slices = slices + 43

    elif tr == 'inter':

        d1 = 0
        d2 = 128

        h1 = 0
        h2 = 128

    # showing image
    slice_img = imgs[0][0][d1:d2, h1:h2, slices]
    axes[0, 0].imshow(slice_img)
    axes[0, 0].axis('off')
    axes[0, 0].set_title(f'img{title[0]}-slice{slices}', fontsize=22)
    # showing gt
    slice_img = imgs[1][0][d1:d2, h1:h2, slices]
    axes[0, 1].imshow(slice_img)
    axes[0, 1].axis('off')
    axes[0, 1].set_title(f'mask{title[0]}-slice{slices}', fontsize=22)
    # showing predicted mask
    slice_img = imgs[2][0][d1:d2, h1:h2, slices]
    axes[0, 2].imshow(slice_img)
    axes[0, 2].axis('off')
    axes[0, 2].set_title(f'prediction{title[0]}-slice{slices}', fontsize=22)
    # showing uncertainty map
    slice_img = imgs[3][0][d1:d2, h1:h2, slices]
    a3 = axes[0, 3].imshow(slice_img, cmap=mlp.cm.hot, interpolation='nearest')
    axes[0, 3].axis('off')
    axes[0, 3].set_title(f'unc_map{title[0]}-slice{slices}', fontsize=22)
    ax0_divider = make_axes_locatable(axes[0, 3])
    cax0 = ax0_divider.append_axes("right", size="7%", pad=0.07)
    cbar1 = fig.colorbar(a3, ax=axes[0, 3], cax=cax0)

    # showing img
    slice_img = imgs[0][1][d1:d2, h1:h2, slices]
    axes[1, 0].imshow(slice_img)
    axes[1, 0].axis('off')
    axes[1, 0].set_title(f'img{title[1]}-slice{slices}', fontsize=22)
    # showing gt
    slice_img = imgs[1][1][d1:d2, h1:h2, slices]
    axes[1, 1].imshow(slice_img)
    axes[1, 1].axis('off')
    axes[1, 1].set_title(f'mask{title[1]}-slice{slices}', fontsize=22)
    # shouwing predicted mask
    slice_img = imgs[2][1][d1:d2, h1:h2, slices]
    axes[1, 2].imshow(slice_img)
    axes[1, 2].axis('off')
    axes[1, 2].set_title(f'prediction{title[1]}-slice{slices}', fontsize=22)
    # showing unc_maps
    slice_img = imgs[3][1][d1:d2, h1:h2, slices]
    a3 = axes[1, 3].imshow(slice_img, cmap=mlp.cm.hot, interpolation='nearest')
    axes[1, 3].axis('off')
    axes[1, 3].set_title(f'unc_map{title[1]}-slice{slices}', fontsize=22)
    ax0_divider = make_axes_locatable(axes[1, 3])
    cax0 = ax0_divider.append_axes("right", size="7%", pad=0.07)
    cbar1 = fig.colorbar(a3, ax=axes[1, 3], cax=cax0)

    if path:
        fig.savefig(path)

    # fig.tight_layout()
    plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()
    plt.close()


##################
## A helper function to show a random list of 2d slices:

rndImg = np.random.choice(64, 2)


def imgshow(img, mask, img_tr=None, mask_tr=None):
    for i in rndImg:
        print(i)

        plt.figure()
        plt.subplot(121)
        plt.imshow(img[:, :, i], cmap='gray')
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(mask[:, :, i], cmap='gray')
        plt.axis("off")

        if img_tr:
            plt.subplot(123)
            plt.imshow(img_tr[:, :, i], cmap='gray')
            plt.axis("off")

            plt.subplot(124)
            plt.imshow(mask_tr[:, :, i], cmap='gray')
            plt.axis("off")


# Create dictionaries for a fast lookup
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.

        ## dump is used to write binary files
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# Binerizing mask
def binerize(mask):
    # this function binerize mask
    mask_b = np.ones_like(mask)
    mask_b = (mask != 0)

    return (mask_b)


def one_hot(img, n_classes=2):
    h, w, d = img.shape

    one_hot = np.zeros((n_classes, h, w, d), dtype=np.float32)

    for i in range(2):
        one_hot[i, :, :, :][img == i] = 1

    return (one_hot)


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


def plotmultiRocCurve(L, tit, svd):
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)

    for df in L:
        plt.plot(df['fdr'], df['tpr'], lw=3)

    textsize = 12
    marker = 5

    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.title('FDR vs TPR')

    plt.grid(b=True, which='major', color='k', linestyle='-')

    plt.grid(b=True, which='minor', color='k', linestyle='--')

    lgd = plt.legend([f'thr_{L[0]}', f'thr_{L[1]}', f'thr_{L[2]}', f'thr_{L[3]}'], \
                     markerscale=marker, prop={'size': textsize, 'weight': 'normal'})

    plt.subplot(1, 2, 2)

    for df in L:
        plt.plot(df['fpr'], df['tpr'], lw=3)

    textsize = 12
    marker = 5

    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plt.title('FPR vs TPR')

    plt.grid(b=True, which='major', color='k', linestyle='-')

    plt.grid(b=True, which='minor', color='k', linestyle='--')

    lgd = plt.legend([f'thr_{L[0]}', f'thr_{L[1]}', f'thr_{L[2]}', f'thr_{L[3]}'], \
                     markerscale=marker, prop={'size': textsize, 'weight': 'normal'})

    fig.tight_layout(pad=3.0)
    # plt.savefig(svd , bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.savefig(svd, bbox_inches='tight')
    plt.show()


def plotDicevsthr(df):
    plt.figure(figsize=(7, 7))
    # plt.fill_between(fpr_list, tpr_list, alpha=0.4)
    plt.plot(df, lw=3)
    plt.xlim(0.1, 0.7)
    plt.ylim(0.7, 1.0)
    plt.xlabel('Uncertainty Threshold', fontsize=15)
    plt.ylabel('Dice', fontsize=15)
    plt.show()


def plotCurves(stats, results_dir=None):
    fig = plt.figure(figsize=(12, 6))

    # for c in stats.keys():
    # plt.plot(stats[c], label=c)

    plt.subplot(1, 2, 1)

    plt.plot(stats['train_loss'], label='train_loss')
    plt.plot(stats['valid_loss'], label='valid_loss')

    textsize = 12
    marker = 5

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.title('NLL')

    plt.grid(b=True, which='major', color='k', linestyle='-')

    plt.grid(b=True, which='minor', color='k', linestyle='--')

    lgd = plt.legend(['train', 'validation'], markerscale=marker,
                     prop={'size': textsize, 'weight': 'normal'})

    ax = plt.gca()

    # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
    #         ax.get_xticklabels() + ax.get_yticklabels()):
    #    item.set_fontsize(textsize)
    #    item.set_weight('normal')

    plt.subplot(1, 2, 2)

    plt.plot(stats['train_dice'], label='train')
    plt.plot(stats['valid_dice'], label='validation')

    textsize = 12
    marker = 5

    plt.xlabel('Epochs')

    plt.ylabel('Dice')

    plt.title('Dice coefficient')

    plt.grid(b=True, which='major', color='k', linestyle='-')

    plt.grid(b=True, which='minor', color='k', linestyle='--')

    lgd = plt.legend(['train', 'validation'], markerscale=marker,
                     prop={'size': textsize, 'weight': 'normal'})

    fig.tight_layout(pad=3.0)
    plt.savefig(results_dir, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.show()


#################################
## Preprocessing

class Resize2(object):

    def __init__(self, in_size):
        self.in_D, self.in_H, self.in_W = in_size

    def __call__(self, vol):
        [depth, height, width] = vol.shape
        scale = [self.in_D * 1.0 / depth, self.in_H * 1.0 / height, self.in_W * 1.0 / width]
        vol = ndimage.interpolation.zoom(vol, scale, order=0)
        return (vol)


class Resize(object):

    def __init__(self, in_size):
        self.in_D, self.in_H, self.in_W = in_size

        assert not self.in_D % 2 and not self.in_H % 2 and not self.in_W % 2, "Input size must be divided by 2!"

    def __call__(self, vol):
        [depth, height, width] = vol.shape
        pad_d = int((self.in_D - depth) / 2)
        pad_h = int((self.in_H - height) / 2)
        pad_w = int((self.in_W - width) / 2)
        vol = np.pad(vol, ((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)), 'constant')
        return (vol)


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, vol):
        vol = (vol - vol.mean()) / vol.std()
        return (vol)


class RandFlip(object):
    def __init__(self, flip_rate):
        self.flip_rate = flip_rate

    def __call__(self, vol):
        if np.random.random_sample() < self.flip_rate:
            return (np.fliplr(vol))
        return (vol)

    # New transformation (resize with padding)


def get_transform(in_size=128):
    in_size = in_size if isinstance(in_size, tuple) else (in_size, in_size, in_size)

    transform = transforms.Compose([Resize(in_size)])

    return (transform)


# Resize with interpolation
def get_transform2(in_size=128):
    in_size = in_size if isinstance(in_size, tuple) else (in_size, in_size, in_size)

    transform = transforms.Compose([Resize2(in_size)])

    return (transform)


################################

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return (group['lr'])


#######################################
## loading datasets

class NPDataSet(Dataset):

    def __init__(self, root_dir, phase, channel_first, transform=None):

        self.root_dir = root_dir

        VolList = sorted([volname for volname in os.listdir(os.path.join(root_dir, phase))])

        MaskList = sorted(
            [maskname for maskname in os.listdir(os.path.join(root_dir, f'{phase}' + '_labels'))])

        # path to each volume
        self.pathvol = sorted([os.path.join(self.root_dir, phase, vol) for vol in VolList])

        # path to each mask
        self.pathmask = sorted(
            [os.path.join(self.root_dir, f'{phase}' + '_labels', mask) for mask in MaskList])

        self.transform = transform

        self.channel_first = channel_first

        self.len = len(VolList)

        assert len(VolList) == len(MaskList)

    def __len__(self):

        return (self.len)

    def __getitem__(self, index):

        # the shape is in the form (depth,highet,width,channel)

        self.vol = np.load(self.pathvol[index])
        self.mask = np.load(self.pathmask[index])

        # cheking to see if its in the form (channel,depth,hight,width)

        assert len(self.vol.shape) == 4

        # Now we transpose the axis if the order is in the shape (depth,hight,width,channel)

        if not self.channel_first:
            self.vol = np.transpose(self.vol, (3, 0, 1, 2))

            self.mask = np.transpose(self.mask, (3, 0, 1, 2))

        # applying transformatios

        if self.transform:
            self.vol = self.transform(self.vol)
            self.mask = self.transform(self.mask)

            ## Here I changed tensors to float32 because I got an error regarding double type
            self.vol = self.vol.astype('float32')
            self.mask = self.mask.astype('float32')

        return (self.vol, self.mask, _)

    ##################################


#### loading Noiseddatset

class NoisedData(Dataset):

    def __init__(self, root_dir, phase, transform=None, rm_out=True, nr=False):

        self.root_dir = root_dir
        self.rm_out = rm_out
        self.nr = nr

        print(f'rm_out in test data:{self.rm_out}')

        dict_list = os.listdir(self.root_dir)

        dict_imgs = [spio.loadmat(os.path.join(self.root_dir, img)) for img in dict_list]

        if phase == 'train':

            self.data = [dic['refData'] for dic in dict_imgs]
            self.gt = [dic['refData_thresholded'] for dic in dict_imgs]

        elif phase == 'test':

            self.data = [dic['testData'] for dic in dict_imgs]
            self.gt = [dic['testData_thresholded'] for dic in dict_imgs]

        self.transform = transform

        self.len = len(self.data)

        assert len(self.data) == len(self.gt)

    def __len__(self):

        return (self.len)

    def rem_outliers(self, vox, nvox=3):

        labels, num = label(vox)

        for i in range(1, num + 1):

            nb_vox = np.sum(vox[labels == i])

            if nb_vox < nvox:
                vox[labels == i] = 0

        return (vox)

    def __getitem__(self, index):

        # the shape is in the form (depth,highet,width,channel)

        self.vol = self.data[index]
        self.mask = self.gt[index]

        # binerizing ground truth
        self.mask = binerize(self.mask)

        if self.rm_out:
            # print(f'rm_out in get_item:{self.rm_out}')

            # print(f'outlier is removed!')
            self.mask = self.rem_outliers(self.mask)

        # cheking to see if its in the form (channel,depth,hight,width)

        # applying transformatios

        if self.transform:
            self.vol = self.transform(self.vol)
            self.mask = self.transform(self.mask)

        if self.nr:
            print(f'Normalized is done!')
            self.vol = Normalize()(self.vol)

        assert len(self.vol.shape) == len(self.mask.shape) == 3
        self.vol = self.vol[np.newaxis, :]
        self.mask = self.mask[np.newaxis, :]

        ## Here I changed tensors to float32 because I got an error regarding double type
        self.vol = self.vol.astype('float32')
        self.mask = self.mask.astype('float32')

        return (self.vol, self.mask)

    ###################################


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
            img_path_vl = val_dir + '/imgs'
            mask_path_vl = val_dir + '/masks'

            os.makedirs(img_path_vl, exist_ok=True)
            os.makedirs(mask_path_vl, exist_ok=True)

            for i, (vol, mask) in enumerate(val_set):
                torch.save(vol, f'{img_path_vl}/img{i}')
                torch.save(mask, f'{mask_path_vl}/mask{i}')

            # save train_set
            img_path_tr = img_path_vl.replace('val_set', 'train_set')
            mask_path_tr = mask_path_vl.replace('val_set', 'train_set')

            os.makedirs(img_path_tr, exist_ok=True)
            os.makedirs(mask_path_tr, exist_ok=True)

            for i, (vol, mask) in enumerate(train_set):
                torch.save(vol, f'{img_path_tr}/img{i}')
                torch.save(mask, f'{mask_path_tr}/mask{i}')

        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                                  drop_last=True, shuffle=True)

        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                                drop_last=True, shuffle=True)

        return (train_loader, val_loader)

    else:

        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                 drop_last=True, shuffle=False)

        return (data_loader)

    ##################################


# Defining the model with MonteCarloDropout

# # input: B,C,H,W,D
# output: B,C,H,W,D

class Conv(nn.Module):

    def __init__(self, in_channel, out_channel, drop=None, bn=True, padding=1, kernel=3,
                 activation=True):

        super(Conv, self).__init__()

        self.conv = nn.Sequential()

        self.conv.add_module('conv', nn.Conv3d(in_channel, out_channel, kernel_size=kernel,
                                               padding=padding))

        if drop is not None:
            self.conv.add_module('dropout', nn.Dropout3d(p=drop))
        if bn:
            self.conv.add_module('bn', nn.BatchNorm3d(out_channel))

        if activation:
            self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return (x)


class DoubleConv(nn.Module):

    def __init__(self, in_channel, out_channel, mid_channel=None, drop=None, drop_mode='all',
                 bn=True, repetitions=2):

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

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2,
                        diffZ - diffZ // 2])

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

    def __init__(self, n_channels, n_classes, n_filters=DEFAULT_FILTERS, depth=DEFAULT_DEPTH,
                 drop=DEFAULT_DROPOUT,
                 drop_center=None, bn=True, bilinear=True):
        super(UNet2, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.bilinear = bilinear
        self.drop = drop
        self.drop_center = drop_center
        self.bn = bn

        # print(f'bilinear in Unet:{self.bilinear}')

        curr_depth = 0
        do_mode = _get_dropout_mode(drop_center, curr_depth, depth, True)
        self.inc = DoubleConv(n_channels, n_filters, drop=drop, drop_mode=do_mode, bn=bn)
        curr_depth += 1
        self.down1 = Down(n_filters, n_filters * 2, drop, drop_center, curr_depth, depth, bn)
        curr_depth += 1
        self.down2 = Down(n_filters * 2, n_filters * 4, drop, drop_center, curr_depth, depth, bn)
        curr_depth += 1
        self.down3 = Down(n_filters * 4, n_filters * 8, drop, drop_center, curr_depth, depth, bn)

        factor = 2 if self.bilinear else 1

        # if self.bilinear:
        # print(f'Hi bilinear is correct!')

        # else:
        # print(f'Hi bilinear is off!')

        self.down4 = Down(n_filters * 8, n_filters * 16 // factor, drop, drop_center, depth, depth,
                          bn)
        curr_depth = 3
        self.up1 = Up(n_filters * 16, n_filters * 8 // factor, drop, drop_center, curr_depth, depth,
                      bn, bilinear)
        curr_depth = 2
        self.up2 = Up(n_filters * 8, n_filters * 4 // factor, drop, drop_center, curr_depth, depth,
                      bn, bilinear)
        curr_depth = 1
        self.up3 = Up(n_filters * 4, n_filters * 2 // factor, drop, drop_center, curr_depth, depth,
                      bn, bilinear)
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

        return (predictions)


###############################
## update learning rate
min_v = 0


def update_lr(lr0, batch_idx, cycle_batch_length, n_sam_per_cycle, optimizer):
    is_end_of_cycle = False

    prop = batch_idx % cycle_batch_length

    pfriction = prop / cycle_batch_length

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


#################################
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

        defaults = dict(lr=lr, weight_decay=weight_decay, temp=temp, addnoise=addnoise,
                        N_train=N_train)

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
                 addnoise=True,
                 epoch_noise=False):

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
                        addnoise=addnoise,
                        epoch_noise=epoch_noise)

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
            epoch_noise = group['epoch_noise']

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

                        buf.mul_(momentum * (group['lr'] / N_train) ** 0.5).add_(d_p,
                                                                                 alpha=1 - dampening)

                    d_p = buf

                if group['addnoise'] and group['epoch_noise']:

                    print(f'noise is injected!')

                    noise = torch.randn_like(p.data).mul_(
                        (temp * group['lr'] * (1 - momentum) / N_train) ** 0.5)

                    p.data.add_(d_p + noise)

                    if torch.isnan(p.data).any(): exit('Nan param')

                    if torch.isinf(p.data).any(): exit('inf param')

                else:

                    p.data.add_(d_p)
        return (loss)


#################################
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
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(
                        group['eps'])

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
                    noise = torch.randn_like(p.data).mul_(
                        (temp * group['lr'] * timestep_factor / N_train) ** 0.5)

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
    ##################################


## Dice score
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

        # prob = F.sigmoid(pred)

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


## Dice loss
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

        # for binary segmentation we must apply sigmoid and for multiclass segmentation
        # we must apply softmax
        # prob = F.softmax(pred,dim=1)

        # prob = F.sigmoid(pred)

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


###############################################

## def main

parser = argparse.ArgumentParser(description='TRAINING SG_MCMC FOR Noise')

# argument experiment
parser.add_argument('--exp', type=int, default=129,
                    help='ID of this expriment!')

parser.add_argument('--gpu', type=int, default=1,
                    help='The id of free gpu for training.')

parser.add_argument('--dr', type=float, default=0.0,
                    help='Dropout rate.')

parser.add_argument('--tr', type=str, default='padd', choices=('inter', 'padd'),
                    help='If we want to use interpolation or padding to resize the imagse.')

parser.add_argument('--tr-evl', type=bool, default=True,
                    help='If we want to evaluate model on train and val set.')

parser.add_argument('--lr-sch', type=str, default='cyclic', choices=(None, 'fixed', 'cyclic'),
                    help='Type of learning rate schedule.')

parser.add_argument('--n-sam-cycle', type=int, default=1,
                    help='Number of samples that we wnat to take in ecah cycle!')

parser.add_argument('--epoch-inject', type=int, default=3,
                    help='The epoch that we want to inject the noise to take samples!')

parser.add_argument('--prior', type=str, default=None, choices=(None, 'norm'),
                    help='if we wnat to use prior or not.')

args = parser.parse_args()

DEFAULT_ALPHA = 0.99
DEFAULT_EPSILON = 1e-7
CLIP_NORM = 0.25


def main(args):
    ### Here name is the name of experiment that I want to perform
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    ## loading hyperparameters
    with open('/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/noise_param.json') as file:
        HPD = json.load(file)

    print(f'Hyperparameters are: {args}')

    exp = args.exp
    gpu = args.gpu
    dr = args.dr
    tr = args.tr
    lr_sch = args.lr_sch
    tr_evl = args.tr_evl
    n_sam_cycle = args.n_sam_cycle
    epoch_inject = args.epoch_inject
    prior = args.prior
    n_epochs = HPD['n_epochs']
    lr0 = HPD['lr0']
    b_size = HPD['b_size']
    n_workers = HPD['n_workers']
    crit = HPD['crit']
    opt = HPD['opt']
    mom = HPD['mom']
    n_filter = HPD['n_filter']
    in_size = HPD['in_size']
    plot = HPD['plot']
    activation = HPD['activation']
    temp = HPD['temp']
    weight_decay = HPD['weight_decay']
    addnoise = HPD['addnoise']
    scale = HPD['scale']
    sampling_start = HPD['sampling_start']
    cycle_length = HPD['cycle_length']
    N_samples = HPD['N_samples']
    save_sample = HPD['save_sample']
    dts = HPD['dts']
    cls = HPD['cls']
    bil = HPD['bil']
    rm_out = HPD['rm_out']
    nr = HPD['nr']
    save_tr_vl = HPD["save_tr_vl"]
    # epoch_noise = HPD['epoch_noise']
    ## storing hyperparmetrs
    HPD['exp'] = exp
    HPD['gpu'] = gpu
    HPD['dr'] = dr
    HPD['tr'] = tr
    HPD['lr_sch'] = lr_sch
    HPD['n_sam_cycle'] = n_sam_cycle
    HPD['epoch_inject'] = epoch_inject
    HPD['prior'] = prior

    print(f'weight_decay: {weight_decay:0.4f}')

    # sanity chcek for hyperparameters
    print(f'parameters for training are:{HPD}')

    # setting device
    device = torch.device(gpu)
    torch.cuda.set_device(device)

    # save parameters as jason file
    with open(f'/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/noise/params/{exp}_noise_param.json',
              'w') as file:
        json.dump(HPD, file, indent=4)

    # gettarsnform2 ===> interpoltaion
    # gettransform ====> zero padding
    if tr == 'inter':
        transform = get_transform2(in_size=in_size)
    elif tr == 'padd':
        transform = get_transform(in_size=in_size)

    if dts == 'noise':
        path_data = '/mnt/home/Masoumeh.Javanbakhat/Baysian/Florian'
        noise_dir = '/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/noise'
        save_dir = '/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/noise/ckpts'
        dataset = NoisedData(path_data, 'train', transform, rm_out, nr)

    os.makedirs(save_dir, exist_ok=True)

    ### Datasets
    # save train and val set
    if save_tr_vl:

        save_train = os.path.join(noise_dir + '/train_set', f'{opt}_{exp}')
        save_val = save_train.replace('train_set', 'val_set')

        print(f'save_train:{save_train}')
        print(f'save_val:{save_val}')

        os.makedirs(save_train, exist_ok=True)
        os.makedirs(save_val, exist_ok=True)

        train_loader, val_loader = get_train_val_loader(dataset, batch_size=b_size,
                                                        num_workers=n_workers, val_size=0.1,
                                                        val_dir=save_val)
    else:

        train_loader, val_loader = get_train_val_loader(dataset, batch_size=b_size,
                                                        num_workers=n_workers, val_size=0.1)

    # defining prior for weights
    # we scale std if we scale loss
    N_train = len(train_loader.dataset)
    print(f'length of train set: {N_train}')

    scaled = N_train ** 0.5

    def normal(m):
        if type(m) == nn.Conv3d:
            print(f'normal prior is applied!')

            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0 * scaled)
            torch.nn.init.normal_(m.bias, mean=0.0, std=1.0 * scaled)

    # scaling weight_decay with number of samples if we have weight decay
    # weight_decay =1 is equal to N(mu=0, sigma=1)
    if weight_decay and scale:
        weight_decay = (weight_decay / N_train)
        print(f'weight decay is scaled with train set size!')

    img, mask = dataset[0]
    print(f'img: {img.shape}, {img.dtype}')
    ## len trainset, valset
    print(f'len dataset :{len(dataset)}')
    print(f'len tarinset:{len(train_loader.dataset)}')
    print(f'len valset:{len(val_loader.dataset)}')

    ### Model
    model = UNet2(n_channels=1, n_classes=cls, n_filters=n_filter, drop=dr, bilinear=bil).to(device)
    print(f'Unet2 with drop {dr} was generated!')

    if prior == 'norm':
        assert scale, "If loss is not scaled, a scaled prior might be harmful"
        model.apply(normal)
        print(f'weights are initialized with scaled normal distribution!')

    ### Optimizer
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr0, weight_decay=weight_decay)


    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=mom, weight_decay=weight_decay)

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

    ### Loss
    if crit == 'dice':
        loss = DiceLoss().to(device)

    elif crit == 'BCrsent':
        loss = nn.BCEWithLogitsLoss().to(device)

    elif crit == 'WBCrsent1':
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(99.)).to(device)

    elif crit == 'Crsent':
        loss = nn.CrossEntropyLoss().to(device)

    print(f'Loss {crit} was used for training!')

    ##### performnace metric
    Dice = DiceCoef()

    #### lr schedule
    n_batch = len(train_loader)
    cycle_batch_length = cycle_length * n_batch
    batch_idx = 0
    print(f'cycle_batch_length: {cycle_batch_length}')

    ##### Trian
    weight_set_samples = []
    sampled_epochs = []

    best_loss = 1000

    loss_total = {'train': [], 'val': []}
    dice_total = {'train': [], 'val': []}
    dice_total2 = {'train': [], 'val': []}

    for epoch in range(n_epochs):

        # print(f'Epoch:{epoch}: {get_lr(optimizer):0.7f}')

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
            total_dice2 = 0

            for j, (vol, mask) in enumerate(dataloader):
                # vol: torch.float32 [b_size,1, h,w,d]
                # mask: torch.float32[b_size,1,h,w,d]
                vol = vol.to(device)
                mask = mask.to(device)

                # weight_fr = torch.ones_like(mask) / 99.0  +  (1.0 - 1.0 / 99.0) * mask

                # print(f'max_we_fr: {weight_fr.max()}, min_we_fr:{weight_fr.min()}')

                # out: torch.float32[b_size,1,h,w,d]
                # probs: torch.float32[b_size, h,w,d]
                # pred: torch.int[b_size,h,w,d]
                out = model(vol)
                probs = F.sigmoid(out)
                pred = (probs > 0.5).float()

                dice_t = Dice(pred.squeeze(), mask.squeeze())
                total_dice += dice_t.item()

                dice_t2 = Dice(probs.squeeze(), mask.squeeze())
                total_dice2 += dice_t2.item()
                # out: torch.float32    [b_size,1,h,w,d]
                # target: torch.int64 [b_size,1,h,w,d]

                if crit == 'BCrsent':
                    target = mask
                    loss_t = loss(out, target)

                elif crit == 'WBCrsent1':
                    target = mask
                    loss_t = loss(out, target)

                # elif crit =='WBCrsent2':
                # traget = mask
                # loss_t = F.binary_cross_entropy(out, target, weight_fr)

                elif crit == 'dice':
                    target = mask
                    loss_t = loss(probs, target)

                elif crit == 'comb':
                    target = mask
                    loss_t1 = nn.BCEWithLogitsLoss()(out, target)
                    loss_t2 = DiceLoss()(probs, target)

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
                        is_end_of_cycle = update_lr(lr0, batch_idx, cycle_batch_length, n_sam_cycle,
                                                    optimizer)

                    loss_t.backward()

                    # clipping loss before updating gradients
                    if lr_sch == 'cyclic':

                        ## clip gradinat by norm to avoide exploding gradiants
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM,
                                                       norm_type=2)

                        if (epoch % cycle_length) + 1 > epoch_inject:

                            optimizer.param_groups[0]['epoch_noise'] = True

                        else:

                            optimizer.param_groups[0]['epoch_noise'] = False

                    optimizer.step()

                    # number of iterations
                    batch_idx += 1

            if scale:
                # devidng loss
                loss_total[phase].append(total_loss / len(dataloader.dataset))
            else:
                # computing loss and dice
                loss_total[phase].append(total_loss / len(dataloader))

            dice_total[phase].append(total_dice / len(dataloader))
            dice_total2[phase].append(total_dice2 / len(dataloader))

            if save_sample:

                if lr_sch == 'cyclic':

                    if epoch >= sampling_start and is_end_of_cycle and phase == 'train':

                        if len(weight_set_samples) >= N_samples:
                            weight_set_samples.pop(0)
                            sampled_epochs.pop(0)

                        weight_set_samples.append(copy.deepcopy(model.state_dict()))
                        sampled_epochs.append(epoch)

                        print(f'sample {len(weight_set_samples)} from {N_samples} was taken!')
                        print(f'End of cycle: {get_lr(optimizer):0.7f}')
                else:

                    if epoch >= sampling_start and epoch % cycle_length == 0 and phase == 'train':

                        if len(weight_set_samples) >= N_samples:
                            weight_set_samples.pop(0)
                            sampled_epochs.pop(0)

                        weight_set_samples.append(copy.deepcopy(model.state_dict()))
                        sampled_epochs.append(epoch)

                        print(f'sample {len(weight_set_samples)} from {N_samples} was taken!')

        # end of epoch
        toc = time.time()
        runtime_epoch = toc - tic

        print(
            'Epoch:%d, loss_train:%0.4f, loss_val:%0.4f, dice_train:%0.4f, dice_val:%0.4f, time:%0.4f seconds' % \
            (epoch, loss_total['train'][epoch], loss_total['val'][epoch], \
             dice_total['train'][epoch], dice_total['val'][epoch], \
             runtime_epoch))

        # saving chcekpoint
        is_best = bool(loss_total['val'][epoch] < best_loss)
        best_loss = loss_total['val'][epoch] if is_best else best_loss
        loss_val = loss_total['val'][epoch]

        checkpoints = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_loss,
            'val_dice_score': dice_total['val'][epoch]}

        torch.save(checkpoints, os.path.join(save_dir, f'{opt}_{exp}'))

    print(f'{len(weight_set_samples)} samples were taken!')

    state = pd.DataFrame({'train_loss': loss_total['train'], 'valid_loss': loss_total['val'],
                          'train_dice': dice_total['train'], 'valid_dice': dice_total['val']})

    state.to_csv(noise_dir + f'/loss/{opt}_exp_{exp}_loss.csv')

    # save model at the end of training
    torch.save({'epoch': epoch,
                'lr': lr0,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'Loss': state}, os.path.join(save_dir, f'{opt}_{exp}_seg3d.pt'))

    if save_sample:
        torch.save(weight_set_samples, os.path.join(save_dir, f'{opt}_{exp}_state_dicts.pt'))
        torch.save(sampled_epochs, os.path.join(save_dir, f'{opt}_{exp}_epochs.pt'))

        # save_object(weight_set_samples, os.path.join(save_dir,f'{opt}_{exp}_state_dicts.pkl'))

    if plot:
        # stats,titl,results_dir=None
        plotCurves(state, noise_dir + f'/lr_curves/{opt}_exp_{exp}_loss.png')

    if tr_evl:

        load_dir = os.path.join(save_dir, f'{opt}_{exp}_seg3d.pt')
        checkpoint = torch.load(load_dir, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)

        train_set = train_loader.dataset
        val_set = val_loader.dataset

        for phase in ['train', 'val']:

            dataset = train_set if phase == 'train' else val_set
            print(f'phase:{phase}')

            dice_t = 0

            for (img, mask) in dataset:
                img = img[np.newaxis, ...]
                mask = mask[np.newaxis, ...]

                img = torch.from_numpy(img).to(device)
                mask = torch.from_numpy(mask).to(device)

                out = model(img)
                prob = F.sigmoid(out)
                pred = (prob > 0.5).floa
                dice_t += Dice(pred.squeeze(), mask.squeeze()).item()

            dice_t /= len(dataset)
            print(f'dice on {phase} set is: {dice_t}')

            np.save(noise_dir + f'/dice_tr_eval/{opt}_exp_{exp}_dice_{phase}.npy', dice_t)

    print(f'model is saved in:')
    print(os.path.join(save_dir, f'{opt}_{exp}_seg3d.pt'))
    print(f'finish training')


#########################

if __name__ == '__main__':
    main(args)

########################

parser2 = argparse.ArgumentParser(description='Evaluation on train and val sets!')

parser2.add_argument('--exp', type=int, default=156,
                     help='ID of this expriment!')

parser2.add_argument('--opt', type=str, default='sghm',
                     help='Which optimizer we use!')

parser2.add_argument('--gpu', type=int, default=0,
                     help='The id of free gpu for training.')

args2 = parser2.parse_args()


def eval_train(exp=None, opt=None, gpu=None):
    # gpu= args2.gpu
    # opt = args2.opt
    # exp = args2.exp

    # setting device
    device = torch.device(gpu)
    torch.cuda.set_device(device)

    # dice
    Dice = DiceCoef()

    noise_dir = '/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/noise'
    save_dir = noise_dir + '/ckpts'

    model = UNet2(n_channels=1, n_classes=1, n_filters=8, drop=0.0, bilinear=1).to(device)

    load_dir = os.path.join(save_dir, f'{opt}_{exp}_seg3d.pt')
    checkpoint = torch.load(load_dir, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)

    save_train = os.path.join(noise_dir + '/train_set', f'{opt}_{exp}')
    save_val = save_train.replace('train_set', 'val_set')

    dice = {'train': 0, 'val': 0}

    for phase in ['train', 'val']:

        len_tr = 14
        len_val = 2

        save_path = save_train if phase == 'train' else save_val
        len_data = 14 if phase == 'train' else 2
        index = [2, 3] if phase == 'train' else [0, 1]

        dice_t = 0

        imgs, masks, preds = [], [], []

        for i in range(len_data):

            img = torch.load(save_path + f'/imgs/img{i}')
            mask = torch.load(save_path + f'/masks/mask{i}')

            img_np = img[np.newaxis, ...]
            mask_np = mask[np.newaxis, ...]

            img = torch.from_numpy(img_np).to(device)
            mask = torch.from_numpy(mask_np).to(device)

            out = model(img)
            prob = F.sigmoid(out)
            pred = (prob > 0.5).float()

            dice_t += Dice(pred.squeeze(), mask.squeeze()).item()

            if i in index:
                imgs.append(img_np.squeeze())
                masks.append(mask_np.squeeze())
                preds.append(pred.cpu().squeeze().numpy())

        dice_t /= len_data

        dice[phase] = dice_t

        print(f'dice on {phase} set is: {dice_t}')

        np.save(noise_dir + f'/dice_tr_eval/{opt}_exp_{exp}_dice_{phase}.npy', dice_t)

        imgs_list = [np.array(imgs), np.array(masks), np.array(preds), np.array(preds)]

        if index:
            slic = 30

            visualize2(noise_dir + f'/results/{opt}_{exp}_pred_{phase}_slice_{slic}.png',
                       imgs=imgs_list,
                       title=index,
                       cols=4,
                       rows=2,
                       plot_size=(24, 10),
                       norm=False,
                       slices=slic,
                       dts=None,
                       tr='padd')

    return (dice)


# if __name__=='__main__':
#    eval_train(args2)


########################
## Uncertainty metrics

def Entropy(p):
    """inputs: softmax probablities of form (tensor): [b_size,n_classes,h,w,d]

       outputs: numpy_array float32: [b_size,h,w,d]
    """
    p = p.cpu().numpy()
    ## we have a vector of softmax probablities : p: [batch_size, num_classes, h,w]
    ## or

    # print(f'p in entropy:{p.min()}, {p.max()}')
    # H: [b_size,h,w]
    H = -(p * np.log(p)).sum(axis=1)

    Hf = H

    p_b = (1 - p)

    # we need this line of code to prevent nan values
    p_b = np.where(p_b == 0, 0.0001, p_b)

    Hb = -(p_b * np.log(p_b)).sum(axis=1)

    Ht = Hf + Hb

    ## Note that if we use a tensor of size p:[batch_size, num_classes,h,w], then meanH gives the mean
    ## of entropy on batch
    ## if p is a tensor of form: [dataset_size, num_classes, h,w], then meanH, returns mean entropy on whole
    ## dataset
    meanH = H.mean(axis=0)

    stdH = H.std(axis=0)

    return (Hf, Hb, Ht)


def Binary_Entropy(p):
    p = p.cpu().numpy()
    p_f = p
    p_b = 1 - p
    p_b = np.where(p_b == 0, 0.0001, p_b)

    H_f = -(p_f * np.log(p_f))
    H_b = -(p_b * np.log(p_b))
    H = -(p_f * np.log(p_f) + p_b * np.log(p_b))

    return (H, H_f, H_b)


def visualize(path=None,
              imgs=None,
              title=None,
              cols=6,
              rows=1,
              plot_size=(20, 20),
              norm=False,
              slices=50,
              dts=None):
    fig, axes = plt.subplots(len(imgs), cols, figsize=plot_size)

    for j, img in enumerate(imgs):

        for i in range(cols):

            slice_img = img[i][:, :, slices]

            if j >= 3:

                if i == 0:
                    a0 = axes[j, i].imshow(slice_img, cmap=cm.coolwarm)
                axes[j, i].imshow(slice_img, cmap=cm.coolwarm)

            else:

                axes[j, i].imshow(slice_img)

            axes[j, i].set_title(f'{title[j]}{i}')

            axes[j, i].axis('off')

        if j >= 3:
            ax1_divider = make_axes_locatable(axes[j, cols - 1])
            cax1 = ax1_divider.append_axes("right", size="7%", pad="4%")
            cbari = fig.colorbar(a0, ax=axes[j, cols - 1], cax=cax1)

    if path:
        fig.savefig(path)

    plt.show()
    plt.close()


#########################
## removing pixels based on uncertainties
def remove_thr(probs, uncs, thersholds):
    """
    probs : np.float32: [len(testset),h,w,d] sigmoid probablites on logits (probablity of positive classes)
    uncs : np.float64: [len(testset),h,w,d]
    thr: a list of float numbers
    return:
           masked_probs_dict: {thr: np.float()}, where np.float() is a numpy array of thersholded probs
    """

    masked_probs_dict = dict.fromkeys(thersholds)

    # outs_mean = np.mean(outputs,axis=(1,2,3))

    # df_mean = dict.fromkeys(thersholds)

    # df_mean['output'] = outs_mean

    for thr in thersholds:

        probs_r = probs.copy()

        uncs_r = uncs.copy()

        masked_probs = np.zeros(probs.shape)

        for i, (prob, unc) in enumerate(zip(probs_r, uncs_r)):
            mask = (unc >= thr)

            prob[mask] = 0

            masked_probs[i, :] = prob

            # masked_outs_mean = np.mean(masked_outs,axis=(1,2,3))
        # df_mean[thr] = masked_outs_mean
        # masked_imgs : np.float64 [len(testset), h,w,d]
        masked_probs_dict[thr] = masked_probs

    print(f'####################################')
    # print(pd.DataFrame(df_mean))
    return (masked_probs_dict)


def dice_after_thersholding(masked_probs, gts, thersholds, b_thrs):
    """gts : list: [len(testset,h,w,d)], float32
       masked_outs: dict: {thr: np.float64(len(testset), h,w,d)}
       thersholds: thersholds that we want to remove uncertain pixels based on them
       b_thr: thersholds that we use to bnrize outputs
    """

    dice_thr = pd.DataFrame(columns=thersholds, index=b_thrs)
    dice_thr.index.name = 'b_thr'
    gts = torch.from_numpy(gts).type(torch.float64)

    for thr in thersholds:

        masked_probs_r = masked_probs.copy()

        masked_prob = masked_probs_r[thr]

        for b_thr in b_thrs:
            temp_pred = (masked_prob >= b_thr).astype(np.float)

            temp_pred_t = torch.from_numpy(temp_pred)

            dice = DiceCoef()(temp_pred_t, gts)

            dice_thr.loc[b_thr][thr] = dice.item()

    print(f'###################################')

    print(dice_thr)

    return (dice_thr)


##########################

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


###################################################

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

    p_unc_ina = iu_t / (ic_t + iu_t)

    p_ac_vs_un = (ac_t + iu_t) / (ac_t + iu_t + ic_t + au_t)

    return (p_acc_con, p_unc_ina, p_ac_vs_un)


###########################################################

def extractPatches(unc_acc_map, window_shape=(4, 4, 4), stride=4):
    """input: im : np.float32: [h,w,d]
              window_shape:    [H,W,D]

       output:patches: np.float32:[nWindow,H,W,D]
    """

    assert len(window_shape) == 3, "shape of winow should be triple!"
    h, w, d = unc_acc_map.shape

    assert not (h % window_shape[0]) and not (w % window_shape[1]) and not (d % window_shape[
        2]), "The shape of image should be divisable by the shape of window!"

    print(window_shape)
    patches = view_as_windows(unc_acc_map, window_shape, step=stride)

    print(patches.shape)

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

    Acc_bin[Acc_arr <= acc_thr] = 0

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


###########################################################

def PAvsPU(Unc_t, Acc_t, acc_thr, thersholds, thr_mood, patch_dim=(4, 4, 4)):
    """inputs:
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
    assert isinstance(thersholds, list)

    if thr_mood == 'max':

        t = np.random.random(1)

        print(f't {t} for setting a thershold')
        print(f'un_min:{thersholds[0]:0.4f}, un_max:{thersholds[1]:0.4f}')
        # un_thr= un_min+(t*(un_max-un_min))
        un_thr = thersholds[0] + (t * (thersholds[1] - thersholds[0]))

    else:

        un_thr = thersholds
        if thr_mood == 'mean':
            print(f'un_mean:{thersholds[0]:0.3f}')

    # we round unc_thrs up to three decimal numbers
    un_thr = list(map(lambda a: round(a, 3), un_thr))
    # print(f'un_thr:{un_thr}')

    L_t = np.zeros((len(un_thr), 7))

    for j, (Unc, Acc) in enumerate(zip(Unc_t, Acc_t)):
        print(f'j:{j}')

        # Unc: np.float64 [h,w] = [128,128]
        # Acc: np.float32 [h,w] = [128,128]

        # Unc_patches :np.float [1024, 4,4]  here nWindows: 1024, each patch is of dimension: [4,4]
        # acc_patches :np.float [1024, 4,4]  here nWindows: 1024, each patch is of dimension: [4,4]
        Unc_patches = extractPatches(Unc, patch_dim, stride=patch_dim[0])

        Acc_patches = extractPatches(Acc, patch_dim, stride=patch_dim[0])

        max_unc = np.max(Unc_patches, axis=(1, 2, 3))
        min_unc = np.min(Unc_patches, axis=(1, 2, 3))

        df_acc_unc = pd.DataFrame(columns=["unc", "acc"], index=np.arange(len(Unc_patches)))

        # Unc_patches_mean : [1024,]
        Unc_patches_mean = Unc_patches.mean(axis=(1, 2, 3))

        # Acc_patches_mean: [1024,]
        Acc_patches_mean = Acc_patches.mean(axis=(1, 2, 3))

        df_acc_unc["unc"] = Unc_patches_mean

        df_acc_unc["acc"] = Acc_patches_mean

        print(f'For patch_window: ')

        df_sorted = df_acc_unc.sort_values(by=['unc'])

        print(df_sorted)

        for i in range(len(un_thr)):
            # here we binerize Acc, Unc Maps

            Acc_bin, Unc_bin = Acc_bin_Unc_bin(Acc_patches_mean, Unc_patches_mean, acc_thr,
                                               un_thr[i])
            # print(f'max_unc_bf_mean:{max_unc[j]}')

            # print(f'min_unc_bf_mean:{min_unc[j]}')
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

    for i in range(len(un_thr)):
        p_acc_con, p_unc_ina, p_ac_vs_un = probabs(L_t[i])

        L_t[i][4] = p_acc_con

        L_t[i][5] = p_unc_ina

        L_t[i][6] = p_ac_vs_un

    df = pd.DataFrame(L_t,
                      columns=['au', 'iu', 'ac', 'ic', 'P(acc|cer)', 'P(uncer|inacc)', 'PAvsPU'],
                      index=np.array(un_thr))
    df.index.name = 'un_thr'
    return (df)


##########################

def b_rates(*a):
    """
    This function compute p(acc|certain), p(uncertain|inacc), p(PacvsPun)
    :params:
            a: is a numpy array : [tn,fp,fn,tp,tpr,fdr,fpr]
    return:
           tpr (sensitivity, recall) = tp/tp+fn
           fdr ()                    = fp/fp+tp
           fpr                       = fp/fp+tn
           prec (precision)          = tp/tp+fp

           f1                        = 2 / (1/prec)+(1/recal)
    """

    tn, fp, fn, tp = a[0][0:4]

    tpr = tp / (tp + fn)

    fdr = fp / (fp + tp)

    fpr = fp / (fp + tn)

    prec = tp / (tp + fp)

    f1 = 2 / ((1 / prec) + (1 / tpr))
    return (tpr, fdr, fpr, prec, f1)


def fpr_fnr(masks_t, probs_t, b_thr):
    """
       masks_t:
       probs_t: np.float32 [n_sample, h,w,d] or a dict:{b_thr:np.float32 [n_sample,h,w,d]}
       pr:     If we are working with probablities of sigmoid or binary masks from unceratinties

    """
    print('Second version of fpr and fnr is used!')

    n_samples = len(masks_t)

    L_t = np.zeros((len(b_thr), 9))

    gt = masks_t.flatten()

    prob = probs_t.flatten()

    # computing roc_auc_score for masks, probabs
    auc = roc_auc_score(gt, prob)

    print(f'AUC for this classifier is {auc:0.4f}')

    # else:

    # print(f'computing tpr and fdr for b_preds of uncertainty removing')

    # preds = [pred.flatten() for pred in probs_t.values()]

    for i in range(len(b_thr)):
        temp_pred = [1 if x >= b_thr[i] else 0 for x in prob]

        conf_mat = confusion_matrix(gt, temp_pred)
        tn, fp, fn, tp = conf_mat.ravel()

        L_t[i][0] += tn
        L_t[i][1] += fp
        L_t[i][2] += fn
        L_t[i][3] += tp

        tpr, fdr, fpr, prec, f1 = b_rates(L_t[i])

        L_t[i][4] = tpr
        L_t[i][5] = fdr
        L_t[i][6] = fpr
        L_t[i][7] = prec
        L_t[i][8] = f1

        ## computing tpr, fpr using sklearn
        # fpr2, tpr2, _ = roc_curve(gt, temp_pred)

    df = pd.DataFrame(L_t, columns=['tn', 'fp', 'fn', 'tp', 'tpr', 'fdr', 'fpr', 'precision', 'f1'],
                      index=np.array(b_thr))
    df.index.name = 'b_thr'

    return (df)


def fpr_fnr_unc(masks_t, probs_t, unc_t, b_thr, unc_thr):
    """
       masks_t:
       probs_t: np.float32 [n_sample, h,w,d] or a dict:{b_thr:np.float32 [n_sample,h,w,d]}
       unc_t :
       b_thr : thershold that we want to binerize the prediction
       unc_thr:thershold for uncertainties

    """
    print('Second version of fpr and fnr is used!')

    n_samples = len(masks_t)

    L_t = np.zeros((len(b_thr), 9))

    gt = masks_t.flatten()

    prob = probs_t.flatten()

    uncf = unc_t.flatten()

    # computing roc_auc_score for masks, probabs
    auc = roc_auc_score(gt, prob)

    print(f'AUC for this classifier is {auc:0.4f}')

    # else:

    # print(f'computing tpr and fdr for b_preds of uncertainty removing')

    # preds = [pred.flatten() for pred in probs_t.values()]

    for i in range(len(b_thr)):

        tp_unc = 0
        tn_unc = 0
        temp_pred = []

        for (x, unc) in zip(prob, uncf):

            if x < b_thr[i] and unc < unc_thr:
                x = 0

            elif x < b_thr[i] and unc > unc_thr:

                x = 1
                tp_unc += 1

            elif x > b_thr[i] and unc < unc_thr:

                x = 1

            elif x > b_thr[i] and unc > unc_thr:
                x = 0
                tn_unc += 1

            temp_pred.append(x)

        conf_mat = confusion_matrix(gt, temp_pred)
        tn, fp, fn, tp = conf_mat.ravel()

        L_t[i][0] += tn
        L_t[i][1] += fp
        L_t[i][2] += fn
        L_t[i][3] += tp

        tpr, fdr, fpr, prec, f1 = b_rates(L_t[i])

        L_t[i][4] = tpr
        L_t[i][5] = fdr
        L_t[i][6] = fpr
        L_t[i][7] = prec
        L_t[i][8] = f1

        ## computing tpr, fpr using sklearn
        # fpr2, tpr2, _ = roc_curve(gt, temp_pred)

    df = pd.DataFrame(L_t, columns=['tn', 'fp', 'fn', 'tp', 'tpr', 'fdr', 'fpr', 'precision', 'f1'],
                      index=np.array(b_thr))
    df.index.name = 'b_thr'

    return (df, tp_unc, tn_unc)


def fpr_fnr4(masks_t, preds_t):
    print('Second version of fpr and fnr is used!')

    n_samples = len(masks_t)

    b_thr = list(np.arange(0.0, 1.1, 0.1))

    L_t = np.zeros((len(b_thr), 3))

    gt = masks_t.flatten()

    pred = preds_t.flatten()

    for i in range(len(b_thr)):
        temp_pred = [1 if x >= b_thr[i] else 0 for x in pred]

        fpr, tpr, _ = roc_curve(gt, temp_pred)

        print(f'{b_thr[i]}: {tpr}, {type(tpr)}, {tpr.shape}')

        print(f'{b_thr[i]}: {fpr}, {type(fpr)}, {fpr.shape}')

        ppv, tpr, _ = precision_recall_curve(gt, temp_pred)

        print(f'{b_thr[i]}: {ppv}, {type(ppv)}, {ppv.shape}')

        fdr = 1 - ppv

        print(f'{b_thr[i]}: {fdr}, {type(fdr)}')

        L_t[i][0] = tpr[0]
        L_t[i][1] = fdr[0]
        L_t[i][2] = fpr[0]

    df = pd.DataFrame(L_t, columns=['tpr', 'fdr', 'fpr'], index=np.array(b_thr))
    df.index.name = 'b_thr'

    return (df)


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'b-', linewidth=2)

    plt.xlabel("Recall", fontsize=16)

    plt.ylabel("Precision", fontsize=16)

    plt.axis([0, 1, 0, 1])

    plt.grid(True)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label='FPR vs TPR')

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel("False Positive Rate (Fall-Out)", fontsize=16)

    plt.ylabel("True Positive Rate (Recall)", fontsize=16)

    plt.grid(True)


def fpr_fnr3(masks_t, probs_t):
    print('Third version of fpr and fnr is used!')

    """
    
       PPV = precision
       FDR = 1- PPV
    
    
    """

    gt = masks_t.flatten()

    prob = probs_t.flatten()

    fpr, tpr, thresholds1 = roc_curve(gt, prob)
    auc_score = roc_auc_score(gt, prob)

    print(f'auc for this classifier is: {auc_score: 0.4f}')
    precision, recall, thresholds2 = precision_recall_curve(gt, prob)

    svd = '/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/noise/results/tprs_crv/sghm_exp_116_300ens.png'

    fig = plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)

    plot_roc_curve(fpr, tpr, label='FPR vs TPR')

    plt.subplot(1, 2, 2)

    plot_precision_vs_recall(precision, recall)

    plt.tight_layout()

    plt.savefig(svd, bbox_inches='tight')
    plt.show()


##########################

def ECE(conf, acc, n_bins=15):
    """
    acc_bm = sigms 1(\hat{y}_i==y_i)/ |b_m|
    conf_bm= sigma \hat{pi} / |b_m|

    acc_bm == conf_bm
    """

    print(f'conf: {conf.shape}')
    print(f'acc:{acc.shape}')

    # here we ravel the tensors that we have
    # conf = conf.ravel()
    # acc = acc.ravel()

    print(f'first implimentation of ece is considered!')

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


#####################################

# conf,acc,n_bins =15

def ECE2(conf, correct, n_bins=15):
    # ravel conf and correct
    conf = conf.ravel()
    correct = correct.ravel()

    correct_list = []
    conf_list = []

    b = np.linspace(start=0, stop=1.0, num=n_bins)
    bins = np.digitize(conf, bins=b, right=True)

    o = 0

    for b in range(n_bins):

        mask = bins == b

        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - conf[mask]))

            correct_list.append(np.mean(correct[mask]))

            conf_list.append(np.mean(conf[mask]))

    ece = o / conf.shape[0]

    return (ece, correct_list, conf_list)


############################
def Brior(probs, gts):
    """
    input : probs: cofidence probablity maps

            gts  : the ground truth masks

    Output: df, brior score for each subject

            brior score total, brior score forground for each subject
    """

    assert isinstance(probs, np.ndarray), 'probs should be numpy array'
    assert isinstance(gts, np.ndarray), 'gts should be numpy array'

    df_brior = pd.DataFrame(index=np.arange(probs.shape[0] + 1),
                            columns=['brior_score', 'brior_score_fr'])

    brior_tot = 0
    brior_fr_tot = 0

    # computing Brior score for each subject
    for i, (prob, lable) in enumerate(zip(probs, gts)):
        df_brior.loc[i]['brior_score'] = np.mean(np.square(prob - lable))

        print(f'brior score for subject {i} is: ')

        print(df_brior.loc[i]['brior_score'])

        fr_s = (lable == 1)

        prob_fr = prob[fr_s]

        lable_fr = lable[fr_s]

        df_brior.loc[i]['brior_score_fr'] = np.mean(np.square(prob_fr - lable_fr))

        print(f'brior score for forground subject {i} is: ')

        print(df_brior.loc[i]['brior_score_fr'])

        brior_fr_tot += df_brior.loc[i]['brior_score_fr']

        brior_tot += df_brior.loc[i]['brior_score']

    df_brior.loc[i + 1]['brior_score'] = brior_tot / probs.shape[0]

    df_brior.loc[i + 1]['brior_score_fr'] = brior_fr_tot / probs.shape[0]

    print(f'Total brior score is:')

    print(df_brior.loc[i + 1]['brior_score'])

    print(f'Total brior score of forground is:')

    print(df_brior.loc[i + 1]['brior_score_fr'])

    return (df_brior)


############################

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


###########################

def remove_thr(preds, uncs, gts, thersholds):
    df = pd.DataFrame(index=thersholds,
                      columns=['dice_mask', 'prd_mean', 'mask_prd_mean', 'ftpr', 'ftnr', 'tpu',
                               'tnu',
                               'dice_fp', 'dice_fn', 'dice_fp_fn', 'ref_vox', 'ref_mis_vox',
                               'dice_add', 'dice_rm'])

    for thr in thersholds:
        print(f'thr:{thr}:')

        uncs = uncs.copy()
        preds_b = preds.copy().astype(np.bool)
        gts_b = gts.copy().astype(np.bool)

        # this is the mask that I want to use
        thersholded_unc = (uncs >= thr)

        tpu, tnu, fpu, fnu, tpu_s, tnu_s, fpu_s, fnu_s, tp, tn, fp, fn = fp_fn(preds_b, gts_b,
                                                                               thersholded_unc,
                                                                               unc=True)

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

        ftp = (tp - tpu_s) / tp
        ftn = (tn - tnu_s) / tn

        df.loc[thr]['ftpr'] = ftp
        df.loc[thr]['ftnr'] = ftn
        df.loc[thr]['tpu'] = tpu_s
        df.loc[thr]['tnu'] = tnu_s

        # correct fp to tn ===> correcting wrong predicted signal
        corrected_preds = preds.copy()
        corrected_preds[fpu] = 0
        corrected_dice_fp = DiceCoef()(corrected_preds, gts)
        # print(f'dice_fp : {corrected_dice_fp}')
        df.loc[thr]['dice_fp'] = corrected_dice_fp.item()

        # correct fn to tp ===> correcting wrong predicted backgrounds
        corrected_preds = preds.copy()
        corrected_preds[fnu] = 1
        corrected_dice_fn = DiceCoef()(corrected_preds, gts)
        # print(f'dice_fn: {corrected_dice_fn}')
        df.loc[thr]['dice_fn'] = corrected_dice_fn.item()

        # correct (fp to tn) and (fn to tp) ====> correcting both wrong predicted signals and backgrounds
        corrected_preds = preds.copy()
        corrected_preds[fpu] = 0
        corrected_preds[fnu] = 1
        corrected_dice_fp_fn = DiceCoef()(corrected_preds, gts)
        # print(f'dice_fp_fn: {corrected_dice_fp_fn}')

        df.loc[thr]['dice_fp_fn'] = corrected_dice_fp_fn.item()

        # correct to foreground :
        corrected_preds = preds.copy()
        corrected_preds[thersholded_unc] = 1
        corrected_add_dice = DiceCoef()(corrected_preds, gts)
        # print(f'dice_for : {corrected_add_dice.item()}')
        df.loc[thr]['dice_add'] = corrected_add_dice.item()

        # correct to background :
        corrected_preds = preds.copy()
        corrected_preds[thersholded_unc] = 0
        corrected_dice = DiceCoef()(corrected_preds, gts)
        df.loc[thr]['dice_rm'] = corrected_dice.item()

        # ratio of voxels that we thershold
        df.loc[thr]['ref_vox'] = (fpu_s + fnu_s) / (tp + tn + fp + fn)

        # ratio of wrong voxels that we went for correction
        df.loc[thr]['ref_mis_vox'] = (fpu_s + fnu_s) / (fp + fn)

    return (df)


#########################

def recall(tp, fn):
    """TP / (TP + FN)"""

    actual_positives = tp + fn

    if actual_positives <= 0:
        return (0)

    return (tp / actual_positives)


def FDR(tp, fp):
    """ FP/ TP+FP"""
    predicted_positives = tp + fp

    if predicted_positives <= 0:
        return (0)

    return (fp / predicted_positives)


def Dice2(tp, fp, fn):
    return ((2 * tp) / (2 * tp + fp + fn))


def roc_unc(probs, gts, uncs, thersholds):
    """ fdr = fp/ actuall(pos) = fp/ fp+tp"""

    b_thrs = [l.round(2) for l in list(np.arange(0.0, 1.1, 0.1))]
    b_thrs.append('retained_voxels')

    df_tpr = pd.DataFrame(columns=b_thrs, index=thersholds)
    df_fdr = pd.DataFrame(columns=b_thrs, index=thersholds)

    for u_thr in thersholds:

        uncs_c = uncs.copy()

        print(f'u_thr: {u_thr}: {uncs_c.mean()}')

        mask = (uncs_c >= u_thr)

        print(f'mask:{mask.mean()}')

        for b_thr in b_thrs[:-1]:
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

            gt_sum = gts_thr.sum()

            pred_thr[mask] = 0
            gts_thr[mask] = 0

            print(f'After masking:')
            print(f'pred_thr: {pred_thr.mean()}')
            print(f'gts_thr: {gts_thr.mean()}')

            gt_mask_sum = gts_thr.sum()

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

        df_tpr.loc[u_thr]['retained_voxels'] = (gt_mask_sum / gt_sum) * 100

    return (df_tpr, df_fdr)


def Unc_stats(unc):
    assert len(unc.shape) == 4
    min_unc = np.min(unc, axis=(1, 2, 3))
    max_unc = np.max(unc, axis=(1, 2, 3))
    mean_unc = np.mean(unc, axis=(1, 2, 3))

    return (min_unc, max_unc, mean_unc)


########################

# dummy functions
def sel_sample(dic, gap=2, cycle_len=3, fresh_sel=False, sel_num=1000):
    cr_cycle = gap * cycle_len

    print(f'current_cycle is {cr_cycle}!')

    sel_epochs = {}

    print(dic)

    for (key, val) in dic.items():

        if fresh_sel:

            if len(sel_epochs.values()) >= sel_num:
                del sel_epochs[next(iter(sel_epochs))]

            sel_epochs[key] = val

        elif val % cr_cycle == 0:

            if len(sel_epochs.values()) >= sel_num:
                del sel_epochs[next(iter(sel_epochs))]

            sel_epochs[key] = val

    print("################")
    print(len(sel_epochs))
    print("################")
    print(f'{len(sel_epochs)} samples with cycle_length {cr_cycle} starting from 15 is generated!')
    print(f'sel_epochs:\n {sel_epochs}')
    return (sel_epochs)


##########################

parser = argparse.ArgumentParser(description='TESTING SG_MCMC FOR Noise')

# argument experiment
parser.add_argument('--exp', type=int, default=130,
                    help='ID of this expriment!')

parser.add_argument('--gpu', type=int, default=5,
                    help='The id of free gpu for training.')

parser.add_argument('--dr', type=float, default=0.0,
                    help='Dropout rate.')

parser.add_argument('--tr', type=str, default='inter', choices=('inter', 'padd'),
                    help='If we want to resize images with intepolation or padding.')

parser.add_argument('--write-exp', type=bool, default=True,
                    help='If we want to write test results in a dataframe.')

parser.add_argument('--sel-epochs', type=bool, default=True,
                    help='if we want to take different epochs for ensembeling.')

parser.add_argument('--gap', type=int, default=3,
                    help='the distance that we want to have between different samples!')

parser.add_argument('--sel-num', type=int, default=33,
                    help='number of samples that we want to take!')

parser.add_argument('--fresh-sel', type=bool, default=False,
                    help='if we want to take fresh samples or not!')

args = parser.parse_args()


## evaluation
def enable_dropout(m):
    if type(m) == nn.Dropout3d:
        m.train()


def test(args):
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    ## loading hyperparameters
    with open('/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/noise_paramt.json') as file:
        HPT = json.load(file)

    exp = args.exp
    gpu = args.gpu
    dr = args.dr
    tr = args.tr
    write_exp = args.write_exp
    sel_epochs = args.sel_epochs
    gap = args.gap
    fresh_sel = args.fresh_sel
    sel_num = args.sel_num
    opt = HPT['opt']
    n_filter = HPT['n_filter']
    activation = HPT['activation']
    crit = HPT['crit']
    in_size = HPT['in_size']
    b_size = HPT['b_size']
    arch = HPT['arch']
    sampler = HPT['sampler']
    Nsamples = HPT['Nsamples']
    dts = HPT['dts']
    metrics = HPT['metrics']
    vis2 = HPT['vis2']
    bil = HPT['bil']
    rm_out = HPT['rm_out']
    sample_weight = HPT['sample_weight']
    HPT['exp'] = exp
    HPT['gpu'] = gpu
    HPT['dr'] = dr
    HPT['tr'] = tr
    HPT['write_exp'] = write_exp
    HPT['sel_epochs'] = sel_epochs
    HPT['gap'] = gap
    HPT['fresh_sel'] = fresh_sel
    HPT['sel_num'] = sel_num

    ## printing hyperparameters
    print(f'Hyperparameters are: {HPT}')

    if args.tr == 'inter':

        transform = get_transform2(in_size=in_size)

    elif args.tr == 'padd':

        transform = get_transform(in_size=in_size)

    if dts == 'noise':
        path_data = '/mnt/home/Masoumeh.Javanbakhat/Baysian/Florian'
        noise_dir = '/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/noise'
        svd = '/mnt/home/Masoumeh.Javanbakhat/Baysian/3D/noise/ckpts'

        print(f'rm_out in test main: {rm_out}')
        testset = NoisedData(path_data, 'test', transform, rm_out)

        print('test on noised dataset!')

    test_loader = get_train_val_loader(testset, val_size=0, batch_size=b_size, num_workers=4)
    print(f'len testset:{len(testset)}')
    print(f'number of batches: {len(test_loader)}')

    load_dir = os.path.join(svd, f'{opt}_{exp}_seg3d.pt')
    print(f'load_dir:{load_dir}')

    device = torch.device(gpu)
    torch.cuda.set_device(device)

    if arch == 'unet2':
        model = UNet2(n_channels=1, n_classes=1, n_filters=n_filter, drop=dr, bilinear=bil)

        print(f'#####################################')
        print(f'Unet2 with drop {dr} was generated!')
        print(f'#####################################')

    checkpoint = torch.load(load_dir, map_location=device)
    epoch = checkpoint['epoch']
    print(f'checkpoints are loaded!')

    if sampler == 'sgmcmc' or sampler == 'sgd':

        # load weights for sgld and sghm samplesr
        # with open(os.path.join(svd,f'{opt}_{exp}_state_dicts.pkl'),'rb') as weights:
        #    weight_set_samples = pickle.load(weights)

        weights_set = torch.load(os.path.join(svd, f'{opt}_{exp}_state_dicts.pt'),
                                 map_location=device)
        sampled_epochs = torch.load(os.path.join(svd, f'{opt}_{exp}_epochs.pt'),
                                    map_location=device)

        print(f'{len(weights_set)} weights are loaded!')
        print(f'model is evaluated for {Nsamples} smaples!')

        assert len(weights_set) == len(sampled_epochs), print(
            'The length of sampled weights and sampled epochs are not equal')

        if sample_weight:

            print(f'sample_weight: {sample_weight}')
            weight_set_samples = []
            idx = sampled_epochs.index(sample_weight)
            weight_set_samples.append(weights_set[idx])
            print(f'weight of epoch {sample_weight} is used for testing!')

        elif sel_epochs:

            # ensembeling on a subset of sampled weights
            sampled_dic = dict(zip(np.arange(len(weights_set)), sampled_epochs))

            cycle_length = sampled_epochs[1] - sampled_epochs[0]

            print(f'cycle_length: {cycle_length}')

            sub_epochs = sel_sample(sampled_dic, gap=gap, cycle_len=cycle_length,
                                    fresh_sel=fresh_sel, sel_num=sel_num)

            print(f'sub_epochs:{sub_epochs}')

            weight_set_samples = weights_set

        else:

            for wieght in (weights_set):

                if len(weights_set) > Nsamples:
                    weights_set.pop(0)
                    sampled_epochs.pop(0)

            weight_set_samples = weights_set
            print(f'{len(weight_set_samples)} samples are used for evaluation!')
            print(f'samples that are used for evaluation:{sampled_epochs}')


    else:

        # load model.state_dict for MC_D
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        print(f'state dict for sampler {sampler} was loaded!')

    print(f'model is evaluated for {Nsamples} samples!')

    print("\n#################################")
    print('Pretrained model is loaded from {%d}th Epoch ' % (epoch))
    print("\n#################################")

    # defining loss
    if crit == 'dice':
        loss = DiceLoss().to(device)

    elif crit == 'BCrsent':
        loss = nn.BCEWithLogitsLoss().to(device)

    elif crit == 'WBCrsent1':

        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(99.)).to(device)

    elif crit == 'Crsent':
        loss = nn.CrossEntropyLoss().to(device)

    # loading test set
    Dice = DiceCoef()

    tic = time.time()
    total_loss = 0
    total_dice = 0

    ch, h, w, d = testset[0][0].shape

    # ent_b =np.zeros((len(testset),h,w,d))
    b_entropy_tot = np.zeros((len(testset), h, w, d))
    f_unc_tot = np.zeros((len(testset), h, w, d))
    b_unc_tot = np.zeros((len(testset), h, w, d))

    out_tot = np.zeros((len(testset), h, w, d))
    preds_tot = np.zeros((len(testset), h, w, d))
    probs_tot = np.zeros((len(testset), h, w, d))

    masks = []
    imgs = []
    n_samples = 0

    with torch.no_grad():

        # model.eval()

        for j, (vol, mask) in enumerate(test_loader):
            # vol: torch.float32 [b_size,1,h,w,d]
            # mask: torch.float32[b_size,1,h,w,d]
            vol = vol.to(device)
            mask = mask.to(device)

            if sampler == 'sgmcmc' or sampler == 'sgd':

                if sel_epochs:

                    out = vol.data.new(len(sub_epochs), b_size, ch, h, w, d)

                    for idx, i in enumerate(sub_epochs.keys()):
                        model.load_state_dict(weight_set_samples[i])

                        model.to(device)

                        out[idx] = model(vol.float())

                else:

                    out = vol.data.new(Nsamples, b_size, ch, h, w, d)

                    for idx, weight_dict in enumerate(weight_set_samples):
                        model.load_state_dict(weight_dict)

                        model.to(device)

                        out[idx] = model(vol.float())

                mean_out = out.mean(dim=0, keepdim=False)
                out = mean_out

            elif sampler == 'mcd':

                model.apply(enable_dropout)

                # we checked sanity check, it was correct
                # for m in model.modules():

                #    if m.__class__.__name__.startswith('Dropout'):

                #        print(f'Dropout is on: {m.training}')

                # mean_out: [b_size,cls,h,w,d]
                out = model.sample_predict(vol, Nsamples).to(device)

                # print(f'out: {out.shape}')
                mean_out = out.mean(dim=0, keepdim=False)
                out = mean_out

            else:

                # sanity check to see that drop out is off, it was correct
                # for m in model.modules():

                # if m.__class__.__name__.startswith('Dropout'):

                # print(f'Dropout is on: {m.training}')

                # out: torch.float32[b_size,1,h,w,d]
                out = model(vol)

            # probs: [b_size,1,h,w,d]
            # preds: [b_size,1,h,w,d]
            probs = F.sigmoid(out).data
            preds = (probs > 0.5).float()

            # computing dice
            preds_np = preds.cpu().squeeze().numpy()
            mask_np = mask.cpu().detach().squeeze().numpy()
            vol_np = vol.cpu().detach().squeeze().numpy()

            if crit == 'BCrsent':
                target = mask
                loss_t = loss(out, target)

            elif crit == 'WBCrsent1':
                target = mask
                loss_t = loss(out, target)

            elif crit == 'dice':
                target = mask
                loss_t = loss(probs, target)

            elif crit == 'comb':
                target = mask
                loss_t1 = nn.BCEWithLogitsLoss()(out, target)
                loss_t2 = DiceLoss()(probs, target)
                loss_t = loss_t1 + loss_t2

            # out: torch.float32    [b_size,1, h,w,d]
            # target: torch.float32 [b_size,1,h,w,d]

            total_loss += loss_t.item()

            imgs.extend(vol_np)
            masks.extend(mask_np)

            probs_tot[n_samples:n_samples + len(vol), :] = probs.cpu().squeeze().numpy()
            out_tot[n_samples:n_samples + len(vol), :] = out.detach().cpu().squeeze().numpy()
            preds_tot[n_samples:n_samples + len(vol), :] = preds.cpu().squeeze().numpy()

            # if sampler:
            b_ent, f_unc, b_unc = Binary_Entropy(probs.squeeze())
            b_entropy_tot[n_samples:n_samples + len(vol), :] = b_ent
            f_unc_tot[n_samples:n_samples + len(vol), :] = f_unc
            b_unc_tot[n_samples:n_samples + len(vol), :] = b_unc

            n_samples += len(vol)

        # computing loss and dice
        total_loss /= len(test_loader)

        # converting img, mask into numpy array
        imgs = np.array(imgs)
        masks = np.array(masks)

        # tr_pos
        correct = (preds_tot == masks)

        # print("\n#################################")
        # print(f'loss_test:{total_loss:0.4f}')
        # print("\n#################################")

        # computing dice for each sample:

        df_dice = pd.DataFrame(index=['dice1', 'dice2', 'tps', 'gts', 'det_sig_per'],
                               columns=np.arange(17))

        dice_t1 = 0
        dice_t2 = 0

        for i, (pred, gt) in enumerate(zip(preds_tot, masks)):
            pred_b = pred.copy().astype(np.bool)
            gt_b = gt.copy().astype(np.bool)

            tp, tn, fp, fn = fp_fn(pred_b, gt_b)

            dice = Dice(pred, gt)
            dice2 = Dice2(tp, fp, fn)

            df_dice.loc['dice1'][i] = dice.item()
            df_dice.loc['dice2'][i] = dice2.item()

            dice_t1 += dice.item()
            dice_t2 += dice2.item()

            df_dice.loc['tps'][i] = tp
            df_dice.loc['gts'][i] = gt_b.sum()

            df_dice.loc['det_sig_per'][i] = (df_dice.loc['tps'][i] / df_dice.loc['gts'][i]) * 100

        total_dice = dice_t1 / preds_tot.shape[0]
        df_dice.loc['dice1'][i + 1] = total_dice
        df_dice.loc['dice2'][i + 1] = dice_t2 / preds_tot.shape[0]

        print("\n###############################")
        print("Dice with thersholding at 0.5 for each sample")
        print(df_dice)
        print("\n#################################")
        print("Dice for whole dataset at 0.5")
        print(f'loss_test:{total_loss:0.4f}, Dice_test:{total_dice:0.4f}')
        print("\n#################################")
        df_dice.to_csv(
            noise_dir + f'/results/dice/{opt}_exp_{exp}_ens_{Nsamples}_thr_{0.5}_dr_{dr}_dice.csv')

        # binary entropy
        min_b_unc, max_b_unc, mean_b_unc = Unc_stats(b_entropy_tot)

        bent_tot_df = pd.DataFrame(
            {'min unc bent': min_b_unc, 'max unc bent': max_b_unc, 'mean b unc': mean_b_unc})
        print(bent_tot_df)
        bent_tot_df.to_csv(
            noise_dir + f'/results/unc/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_unc.csv', index=True)

        print("\n###############################")
        print(f'Visulaizing preds')

        for pred in preds_tot:
            print(f'{pred[30, :, :].min()}, {pred[30, :, :].max()}')

        print(f'######################################')
        print(f'Computing tpr and fdr')

        # save images, masks and entropy masks for visualization perpouses
        np.save(
            noise_dir + f'/results/test_imgs/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_test_imgs.npy',
            imgs)
        np.save(
            noise_dir + f'/results/test_masks/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_test_masks.npy',
            masks)
        np.save(
            noise_dir + f'/results/test_preds/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_test_preds.npy',
            preds_tot)
        np.save(
            noise_dir + f'/results/test_ent_maps/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_test_ents.npy',
            b_unc_tot)
        np.save(noise_dir + f'/results/probs/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_probs.npy',
                probs_tot)

        if vis2:
            img = []
            mask = []
            pred = []
            ent = []

            for ind in vis2:
                img.append(imgs[ind])
                mask.append(masks[ind])
                pred.append(preds_tot[ind])
                ent.append(b_entropy_tot[ind])

            img = np.array(img)

            mask = np.array(mask)
            pred = np.array(pred)
            ent = np.array(ent)
            imgs_list = [img, mask, pred, ent]

            title = vis2

        print('######################################')

        if metrics:

            if 'rm_unc' in metrics:
                print(f'removing uncs in predicted mask')

                rm_thr = [l.round(2) for l in list(np.arange(0.1, 0.8, 0.1))]

                print(rm_thr)

                df = remove_thr(preds_tot, b_entropy_tot, masks, rm_thr)

                df.to_csv(
                    noise_dir + f'/results/rm_unc/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_rm_unc.csv',
                    index=True)

                print(df)

            if 'roc_unc' in metrics:
                thr_roc_unc = [0.4, 0.5, 0.6, 0.69, 0.7]

                print(f'Computing tpr and fdr in thersholded uncs')

                df_tpr, df_fdr = roc_unc(probs_tot, masks, b_entropy_tot, thr_roc_unc)

                df_tpr.to_csv(
                    noise_dir + f'/results/roc_unc/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_tpr_unc.csv', \
                    index=True)

                df_fdr.to_csv(
                    noise_dir + f'/results/roc_unc/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_fdr_unc.csv', \
                    index=True)

            if 'pacvpu' in metrics:
                thersholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

                df = PAvsPU(Unc_t=b_entropy_tot, Acc_t=correct, acc_thr=0.5, thersholds=thersholds, \
                            thr_mood='interval', patch_dim=(1, 1, 1))

                df.to_csv(
                    noise_dir + f'/results/pacvpu/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_pavpu.csv',
                    index=True)

                print(df)

            if 'ece' in metrics:

                pred_prob = np.zeros(probs_tot.shape)

                pred_prob = np.where(probs_tot > 1 - probs_tot, probs_tot, 1 - probs_tot)

                ece, acc, conf = ECE(pred_prob, correct)
                print('####### ECE #######\n')
                print(f'ECE:{ece.item():0.4f}')

                np.save(noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_ece.npy',
                        ece)
                np.save(noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_acc.npy',
                        acc)
                np.save(noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_conf.npy',
                        conf)

                # calibration for true positive

                print(f'Computing calibration for true positives!')
                preds_b = preds_tot.copy().astype(np.bool)
                gts_b = masks.copy().astype(np.bool)

                tps = np.logical_and(preds_b, gts_b)
                prob_tps = np.where(probs_tot > 0.5, probs_tot, 0)

                ece_tps, acc_tps, conf_tps = ECE(prob_tps, tps)

                np.save(
                    noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_ece_tps.npy',
                    ece_tps)
                np.save(
                    noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_acc_tps.npy',
                    acc_tps)
                np.save(
                    noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_conf_tps.npy',
                    conf_tps)

                if 'ece_s' in metrics:

                    print(f'computing ece for subjects!')

                    idx = [2, 4, 6, 8]

                    for i in idx:
                        print(f'ece for subject {i}')

                        ece_i, acc_i, conf_i = ECE(pred_prob[i], correct[i])
                        print(f'ece subject {i}:{ece_i}')
                        print(f'acc subjcet {i}:{acc_i}')
                        print(f'conf subject{i}:{conf_i}')

                        np.save(
                            noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_sub_{i}_ece_s.npy',
                            ece_i)
                        np.save(
                            noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_sub_{i}_acc_s.npy',
                            acc_i)
                        np.save(
                            noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_sub_{i}_conf_s.npy',
                            conf_i)

                        print(f'ece for tp of subject {i}')
                        ece_tps_i, acc_tps_i, conf_tps_i = ECE(prob_tps[i], tps[i])

                        np.save(
                            noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_sub_{i}_ece_tps_s.npy',
                            ece_tps_i)
                        np.save(
                            noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_sub_{i}_acc_tps_s.npy',
                            acc_tps_i)
                        np.save(
                            noise_dir + f'/results/ece/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_sub_{i}_conf_tps_s.npy',
                            conf_tps_i)

            if 'brior' in metrics:
                # here I changed all probablities to probablities of positive class
                # pred_prob = np.where(probs_tot>1-probs_tot,probs_tot,1-probs_tot)

                brior_score = Brior(probs_tot, masks)

                brior_score.to_csv(
                    noise_dir + f'/results/brior/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_brior.csv',
                    index=True)

            if 'unc_err' in metrics:

                thr_unc_err = [l.round(2) for l in list(np.arange(0.1, 0.8, 0.1))]

                err_tot = (preds_tot != masks)

                dic_tot = dict.fromkeys(thr_unc_err)

                df_unc = pd.DataFrame(columns=thr_unc_err, index=np.arange(16))

                for thr in thr_unc_err:

                    unc_thr = (b_entropy_tot >= thr).astype(np.float)

                    Dice_unc = Dice(err_tot, unc_thr)

                    dic_tot[thr] = Dice_unc.item()

                    for i, (err_s, unc_s) in enumerate(zip(err_tot, unc_thr)):
                        dice_s = Dice(err_s, unc_s)

                        df_unc.loc[i][thr] = dice_s.item()

                print(f'Dice is: {dic_tot}')

                print(f'Dice for each sample is:')

                print(df_unc)

                df_dic = pd.DataFrame(dic_tot, index=["dice"])

                df_dic.to_csv(
                    noise_dir + f'/results/unc_err/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_unc_err_tot.csv',
                    index=True)
                df_unc.to_csv(
                    noise_dir + f'/results/unc_err/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_unc_err.csv',
                    index=True)

            if 'unc_acc' in metrics:

                thr_unc_err = [l.round(2) for l in list(np.arange(0.1, 0.8, 0.1))]

                df_unc_acc = pd.DataFrame(index=["acc", "acc_f", "acc_b"], columns=thr_unc_err)

                df_unc_acc_s = pd.DataFrame(columns=thr_unc_err, index=np.arange(16))

                for thr in thr_unc_err:

                    unc_thr = (b_entropy_tot >= thr).astype(np.float)

                    Dice_acc = Dice(correct, unc_thr)

                    # looking at acc of predictions
                    df_unc_acc.loc["acc"][thr] = Dice_acc.item()

                    for i, (acc_s, unc_s) in enumerate(zip(correct, unc_thr)):
                        dice_s = Dice(acc_s, unc_s)

                        df_unc_acc_s.loc[i][thr] = dice_s.item()

                # saving dataframes
                df_unc_acc.to_csv(
                    noise_dir + f'/results/unc_acc/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_unc_acc.csv')
                df_unc_acc.to_csv(
                    noise_dir + f'/results/unc_acc/{opt}_exp_{exp}_ens_{Nsamples}_dr_{dr}_unc_acc_s.csv')

        if vis2:

            if tr == 'inter':

                slic = 100

            elif tr == 'padd':

                slic = 34

            visualize2(
                noise_dir + f'/results/{opt}_{exp}_pred_ens_{Nsamples}_dr_{dr}_slice_{slic}.png',
                imgs=imgs_list,
                title=title,
                cols=4,
                rows=2,
                plot_size=(24, 10),
                norm=False,
                slices=slic,
                dts=dts,
                tr=tr)

        print('######################################')
        # print(f'Evaluation on train and validation set')
        # dice_train = eval_train(exp=exp, opt=opt, gpu=gpu)

        # saving results in a dataframe
        if write_exp:

            data = {
                'exp': [exp],
                'in_size': [in_size],
                'opt': [opt],
                'dr': [dr],
                'filter': [n_filter],
                'tr': [tr],
                'n_ensembel': [Nsamples],
                'nll': [round(total_loss, 4)],
                'dice': [round(total_dice, 4)]}
            # 'dice_train':[round(dice_train['train'],4)],
            # 'dice_val':[round(dice_train['val'],4)]}

            # ./Baysian_Seg/Results/
            csv_path = noise_dir + f'/results/nll/run_sweeps_test.csv'

            if os.path.exists(csv_path):

                sweeps_df = pd.read_csv(csv_path)
                sweeps_df = sweeps_df.append(
                    pd.DataFrame.from_dict(data), ignore_index=True).set_index('exp')

            else:

                sweeps_df = pd.DataFrame.from_dict(data).set_index('exp')

            # save experiment metadata csv file
            sweeps_df.to_csv(csv_path)

        # if __name__=='__main__':

#    test(args)
