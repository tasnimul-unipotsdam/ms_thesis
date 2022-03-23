import glob
import os
import nibabel as nib
import numpy as np
from glob import glob
from scipy import ndimage


def read_nifti_file(filepath):
    file = nib.load(filepath)
    file = file.get_fdata()
    return file


def normalize(vol):
    vol = (vol - vol.mean()) / vol.std()
    return vol


def resize_volume(vol):
    """Resize across z-axis"""
    # Set the desired depth

    desired_width = 128
    desired_height = 128
    desired_depth = 128
    # Get current depth

    current_width = vol.shape[0]
    current_height = vol.shape[1]
    current_depth = vol.shape[-1]
    # Compute depth factor

    width = current_width / desired_width
    height = current_height / desired_height
    depth = current_depth / desired_depth

    width_factor = 1 / width
    height_factor = 1 / height
    depth_factor = 1 / depth

    # Resize across z-axis
    vol = ndimage.zoom(vol, (width_factor, height_factor, depth_factor), order=0)
    return vol


def volume_process(path):
    volume = read_nifti_file(path)
    volume = resize_volume(volume)
    volume = normalize(volume)
    return volume


def binary_mask(mask):
    mask_binary = np.where(mask > 0, 1, 0)
    return mask_binary


def mask_process(path):
    mask = read_nifti_file(path)
    mask = resize_volume(mask)
    mask = binary_mask(mask)
    return mask


def get_item_list():
    path = ".//brain_data"

    _image_path = sorted(glob(os.path.join(path, "*", "*_t2.nii.gz")))
    _mask_path = sorted(glob(os.path.join(path, "*", "*_seg.nii.gz")))

    train_images_list = _image_path[:1000]
    train_masks_list = _mask_path[:1000]

    val_images_list = _image_path[1000:]
    val_masks_list = _mask_path[1000:]

    print(len(train_images_list), len(train_masks_list), len(val_images_list), len(val_masks_list))

    # print(_image_path)

    return train_images_list, train_masks_list, val_images_list, val_masks_list


def nifti_numpy(axis=1):
    train_images_list, train_masks_list, val_images_list, val_masks_list = get_item_list()
    print("start process")

    train_image_process = np.array([volume_process(path) for path in train_images_list]).astype(
        'float32')
    print("process 2")

    train_mask_process = np.array([mask_process(path) for path in train_masks_list]).astype('float32')

    print("process 3")

    val_image_process = np.array([volume_process(path) for path in val_images_list]).astype(
        'float32')

    print("process 4")

    val_mask_process = np.array([mask_process(path) for path in val_masks_list]).astype('float32')

    print("expansion")

    train_image = np.expand_dims(train_image_process, axis=axis)
    # train_mask = np.expand_dims(train_mask_process, axis=axis)

    valid_image = np.expand_dims(val_image_process, axis=axis)
    # valid_mask = np.expand_dims(val_mask_process, axis=axis)

    print("saving")

    np.save(".//numpy_data/train_image_t2", train_image)
    # np.save(".//numpy_data/train_label_t2", train_mask)

    np.save(".//numpy_data/val_image_t2", valid_image)
    # np.save(".//numpy_data/val_label_t2", valid_mask)

    print("end")

    print(train_image.shape)
    # print(train_mask.shape)

    print(valid_image.shape)
    # print(valid_mask.shape)

    # print(np.unique(train_mask))
    # print(np.unique(valid_mask))

    return train_image, valid_image


if __name__ == '__main__':
    nifti_numpy()
