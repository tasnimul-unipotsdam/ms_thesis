import glob
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from scipy import ndimage


def rot_blur(image, image_np):
    # image_np = ndimage.rotate(image_np, 90, reshape=False)
    image_np = gaussian_filter(image_np, sigma=14)

    image_nifti = nib.Nifti1Image(image_np, image.affine)
    return image_nifti


def write_images(file_path, image_nifti):
    names = os.path.basename(file_path).split('.')
    dest_file_name = names[0] + '.' + names[1] + "." + names[2]
    nib.save(image_nifti, os.path.join(dest_path, dest_file_name))


if __name__ == '__main__':
    data_path = ".//nifti_data//test_image"
    dest_path = ".//nifti_data/test_image_blur14"

    files = glob.glob(os.path.join(data_path, '*'))

    for file in files:
        image = nib.load(file)
        image_np = nib.load(file, mmap=False).get_fdata(caching='unchanged')

        img = rot_blur(image, image_np)
        write_images(file_path=file, image_nifti=img)