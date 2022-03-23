import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader


def numpy_data_loader():
    test_t2 = np.load(".//data/volume_t2.npy")
    test_mask = np.load(".//data/mask_t2.npy")

    return test_t2, test_mask


def torch_test_data_loader(batch_size=2, workers=4):
    test_t2, test_mask = numpy_data_loader()

    test_image_tensor = torch.from_numpy(test_t2)
    test_mask_tensor = torch.from_numpy(test_mask)

    test_dataset = TensorDataset(test_image_tensor, test_mask_tensor)

    print(test_image_tensor.shape)
    print(test_mask_tensor.shape)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=workers,
                             pin_memory=False,
                             prefetch_factor=1)

    return test_loader


if __name__ == '__main__':
    print("data_loader")
    # numpy_data_loader()
    torch_test_data_loader()
