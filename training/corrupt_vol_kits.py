import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

number_of_rect = 11


def rect(image, image_np):
    x = np.random.randint(0, 160, number_of_rect)
    y = np.random.randint(0, 512, number_of_rect)
    z = np.random.randint(0, 512, number_of_rect)

    a = 80
    b = 80
    c = 80

    x_start = np.maximum(x - a, 0)
    y_start = np.maximum(y - b, 0)
    z_start = np.maximum(z - c, 0)

    x_end = np.minimum(x + a, image.shape[0])
    y_end = np.minimum(y + b, image.shape[1])
    z_end = np.minimum(z + c, image.shape[2])

    for i in range(0, number_of_rect):
        image_np[x_start[i]:x_end[i], y_start[i]:y_end[i], z_start[i]:z_end[i]] = 0

    image_nifti = nib.Nifti1Image(image_np, image.affine)
    return image_nifti


def write_images(file_path, image_nifti):
    # This name is hardcoded as this file will be saved in separate unique directory
    dest_file_name = f"{os.path.basename(Path(file_path).parent.absolute())}_imaging_corrupt.nii.gz"
    dest_path = os.path.join(Path(file_path).parent.absolute(), dest_file_name)
    nib.save(image_nifti, dest_path)


if __name__ == '__main__':
    data_path = ".//dataset"
    files = sorted(glob.glob(os.path.join(data_path, "test", "*", "imaging.nii.gz")))

    for file in tqdm(files):
        try:
            image = nib.load(file)
            image_np = nib.load(file, mmap=False).get_fdata(caching='unchanged')

            img = rect(image, image_np)
            write_images(file, image_nifti=img)
        except Exception as ex:
            print(f"ERROR: Can't load corrupted file. File path: {file}")

