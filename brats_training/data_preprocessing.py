import glob
import os
import nibabel as nib
import numpy as np
from glob import glob
from scipy import ndimage
from sklearn.model_selection import train_test_split


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


def volume_numpy(axis=1):
    path = ".//brain_data"
    volume_path = sorted(glob(os.path.join(path, "*", "*_t2.nii.gz")))

    volume_path = volume_path[:250]

    print("creating numpy array of volume")
    volume = np.array([volume_process(path) for path in volume_path]).astype('float32')
    volume = np.expand_dims(volume, axis=axis)
    print("saving volume")
    np.save(".//data/volume_t2", volume)

    print(volume.shape)
    print(np.unique(volume))

    return volume


def mask_numpy(axis=1):
    path = ".//brain_data"
    mask_path = sorted(glob(os.path.join(path, "*", "*_seg.nii.gz")))

    mask_path = mask_path[:250]

    print("creating numpy array of masks")
    mask = np.array([mask_process(path) for path in mask_path]).astype('float32')
    mask = np.expand_dims(mask, axis=axis)
    print("saving masks")
    np.save(".//data/mask_t2", mask)

    print(mask.shape)
    print(np.unique(mask))

    return mask


def train_test_val_split():
    volume = np.load(".//data/volume_flair.npy")
    mask = np.load(".//data/mask.npy")

    train_flair, test_flair, train_mask, test_mask = train_test_split(volume, mask, test_size=0.2,
                                                                      random_state=1, shuffle=True)

    train_flair, val_flair, train_mask, val_mask = train_test_split(train_flair, train_mask,
                                                                    test_size=0.25, random_state=1,
                                                                    shuffle=True)

    print(train_flair.shape, train_mask.shape, val_flair.shape, val_mask.shape, test_flair.shape,
          test_mask.shape)
    np.save(".//data/train_flair", train_flair)
    np.save(".//data/train_mask", train_mask)
    np.save(".//data/val_flair", val_flair)
    np.save(".//data/val_mask", val_mask)
    np.save(".//data/test_flair", test_flair)
    np.save(".//data/test_mask", test_mask)

    return train_flair, train_mask, val_flair, val_mask, test_flair, test_mask


if __name__ == '__main__':
    volume_numpy()
    # mask_numpy()
    # train_test_val_split()
