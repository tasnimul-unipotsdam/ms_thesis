import glob
import os
from glob import glob

import nibabel as nib
import numpy as np
from scipy import ndimage


def read_nifti_file(filepath):
    file = nib.load(filepath)
    file = file.get_fdata()
    return file


def resize_volume(vol):
    """Resize across z-axis"""
    # Set the desired depth
    desired_height = 128
    desired_width = 128
    desired_depth = 128
    # Get current depth
    current_height = vol.shape[0]
    current_width = vol.shape[1]
    current_depth = vol.shape[-1]
    # Compute depth factor
    height = current_height / desired_height
    width = current_width / desired_width
    depth = current_depth / desired_depth

    height_factor = 1 / height
    width_factor = 1 / width
    depth_factor = 1 / depth

    # Resize across z-axis
    vol = ndimage.zoom(vol, (height_factor, width_factor, depth_factor), order=0, mode="nearest")
    return vol


def normalize(vol):
    vol = (vol - vol.mean()) / vol.std()
    return vol


def expand(volume, axis):
    volume = np.expand_dims(volume, axis=axis)
    return volume


def one_hot_encode(img):
    label0 = (img == 0)
    label1 = np.logical_or(np.logical_or(img == 1, img == 2), img == 3)
    label2 = np.logical_or(img == 2, img == 3)
    one_hot = np.stack((label0, label1, label2), axis=0)
    return one_hot


def image_process(path):
    volume = read_nifti_file(path)
    volume = resize_volume(volume)
    volume = normalize(volume)
    volume = expand(volume, axis=0)
    return volume


def mask_process(path):
    volume = read_nifti_file(path)
    volume = resize_volume(volume)
    volume = one_hot_encode(volume)
    return volume


def train_data():
    path = ".//dataset"
    volume_path = sorted(glob(os.path.join(path, "train", "*", "imaging.nii.gz")))
    mask_path = sorted(glob(os.path.join(path, "train", "*", "aggregated_MAJ_seg.nii.gz")))

    print("vol")
    image_list = []
    for path in volume_path:
        try:
            image_list.append(image_process(path))
        except Exception as ex:
            print(path)

    images = np.array(image_list).astype('float32')

    print("mask")
    mask_list = []
    for path in mask_path:
        try:
            mask_list.append(mask_process(path))
        except Exception as ex:
            print(path)

    masks = np.array(mask_list).astype('float32')

    np.save('.//numpy_data/train_image', images)
    np.save('.//numpy_data/train_mask', masks)

    print(images.shape)
    print(masks.shape)


def valid_data():
    path = ".//dataset"
    volume_path = sorted(glob(os.path.join(path, "validation", "*", "imaging.nii.gz")))
    mask_path = sorted(glob(os.path.join(path, "validation", "*", "aggregated_MAJ_seg.nii.gz")))

    print("vol")
    image_list = []
    for path in volume_path:
        try:
            image_list.append(image_process(path))
        except Exception as ex:
            print(path)

    images = np.array(image_list).astype('float32')

    print("mask")
    mask_list = []
    for path in mask_path:
        try:
            mask_list.append(mask_process(path))
        except Exception as ex:
            print(path)

    masks = np.array(mask_list).astype('float32')

    np.save('.//numpy_data/val_image', images)
    np.save('.//numpy_data/val_mask', masks)

    print(images.shape)
    print(masks.shape)


def test_data():
    path = ".//dataset"
    volume_path = sorted(glob(os.path.join(path, "test", "*", "imaging.nii.gz")))
    mask_path = sorted(glob(os.path.join(path, "test", "*", "aggregated_MAJ_seg.nii.gz")))

    print("vol")
    image_list = []
    for path in volume_path:
        try:
            image_list.append(image_process(path))
        except Exception as ex:
            print(path)

    images = np.array(image_list).astype('float32')

    print("mask")
    mask_list = []
    for path in mask_path:
        try:
            mask_list.append(mask_process(path))
        except Exception as ex:
            print(path)
    masks = np.array(mask_list).astype('float32')

    np.save('.//numpy_data/test_image', images)
    np.save('.//numpy_data/test_mask', masks)

    print(images.shape)
    print(masks.shape)


def test_image_ood():
    path = ".//dataset"
    volume_path = sorted(glob(os.path.join(path, "test", "*", "*_imaging_corrupt.nii.gz")))

    print("vol")
    image_list = []
    for path in volume_path:
        try:
            image_list.append(image_process(path))
        except Exception as ex:
            print(path)

    images = np.array(image_list).astype('float32')

    np.save('.//numpy_data/test_image_ood', images)

    print(images.shape)


if __name__ == '__main__':
    # train_data()
    # valid_data()
    # test_data()
    test_image_ood()
