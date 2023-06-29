import glob
import os
from glob import glob

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.utils.class_weight import compute_class_weight
import torch


def read_nifti_file(filepath):
    file = nib.load(filepath)
    file = file.get_fdata()
    return file


def resize_volume(vol):
    """Resize across z-axis"""
    # Set the desired depth
    desired_height = 128
    desired_width = 128
    desired_depth = 64
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


def one_hot_encode(img, n_classes):

    h, w, d = img.shape

    one_hot = np.zeros((n_classes, h, w, d), dtype=np.float32)

    for i, unique_value in enumerate(np.unique(img)):
        one_hot[i, :, :, :][img == unique_value] = 1

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
    volume = one_hot_encode(volume, n_classes=16)
    return volume


def train_data():
    path = ".//dataset"
    volume_path = sorted(glob(os.path.join(path, "train_image", "*.nii.gz")))
    mask_path = sorted(glob(os.path.join(path, "train_label", "*.nii.gz")))

    images = np.array([image_process(path) for path in volume_path]).astype('float32')
    masks = np.array([mask_process(path) for path in mask_path]).astype('float32')

    np.save('.//numpy_data/train_image', images)
    np.save('.//numpy_data/train_mask', masks)

    print(images.shape)
    print(masks.shape)


def valid_data():
    path = ".//dataset"
    volume_path = sorted(glob(os.path.join(path, "validation_image", "*.nii.gz")))
    mask_path = sorted(glob(os.path.join(path, "validation_label", "*.nii.gz")))

    images = np.array([image_process(path) for path in volume_path]).astype('float32')
    masks = np.array([mask_process(path) for path in mask_path]).astype('float32')

    np.save('.//numpy_data/val_image', images)
    np.save('.//numpy_data/val_mask', masks)

    print(images.shape)
    print(masks.shape)


def test_data():
    path = ".//dataset"
    volume_path = sorted(glob(os.path.join(path, "test_image", "*.nii.gz")))
    mask_path = sorted(glob(os.path.join(path, "test_label", "*.nii.gz")))

    images = np.array([image_process(path) for path in volume_path]).astype('float32')
    masks = np.array([mask_process(path) for path in mask_path]).astype('float32')

    np.save('.//numpy_data/test_image', images)
    np.save('.//numpy_data/test_mask', masks)

    print(images.shape)
    print(masks.shape)


def test_image_ood():
    path = ".//dataset"
    volume_path = sorted(glob(os.path.join(path, "MRI_IMAGE", "*.nii.gz")))
    mask_path = sorted(glob(os.path.join(path, "MRI_LABEL", "*.nii.gz")))

    images = np.array([image_process(path) for path in volume_path]).astype('float32')
    masks = np.array([mask_process(path) for path in mask_path]).astype('float32')

    np.save('.//numpy_data/test_image_ood', images)
    np.save('.//numpy_data/test_mask_ood', masks)

    print(images.shape)
    print(masks.shape)


def mask_process_compute_class_weight(path):
    volume = read_nifti_file(path)
    volume = resize_volume(volume)
    return volume


def compute_class_weights():
    path = ".//dataset"
    mask_path = sorted(glob(os.path.join(path, "train_label", "*.nii.gz")))
    print(len(mask_path))

    masks = np.array([mask_process_compute_class_weight(path) for path in mask_path]).astype('float32').flatten()
    print(masks.shape)
    classes = np.unique(masks, return_counts=True)
    print(classes)
    # class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(masks), y=masks)
    # class_weights = torch.tensor(class_weights, dtype=torch.float32)
    # print(class_weights)


if __name__ == '__main__':
    train_data()
    valid_data()
    test_data()
    test_image_ood()
    # compute_class_weights()
