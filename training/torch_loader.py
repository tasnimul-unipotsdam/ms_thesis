import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader


def train_data_loader(batch_size=2, workers=4):
    train_image = np.load(".//numpy_data/train_image.npy")
    train_mask = np.load(".//numpy_data/train_mask.npy")

    val_image = np.load(".//numpy_data/val_image.npy")
    val_mask = np.load(".//numpy_data/val_mask.npy")

    train_image_tensor = torch.from_numpy(train_image)
    train_mask_tensor = torch.from_numpy(train_mask)

    valid_image_tensor = torch.from_numpy(val_image)
    valid_mask_tensor = torch.from_numpy(val_mask)

    print(train_image_tensor.shape)
    print(valid_image_tensor.shape)
    print(train_mask_tensor.shape)
    print(valid_mask_tensor.shape)

    train_dataset = TensorDataset(train_image_tensor, train_mask_tensor)
    valid_dataset = TensorDataset(valid_image_tensor, valid_mask_tensor)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers,
                              pin_memory=True,
                              prefetch_factor=1)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=workers,
                              pin_memory=False,
                              prefetch_factor=1)

    return train_loader, valid_loader


def test_data_loader(batch_size=1, workers=4):
    test_image = np.load(".//numpy_data/test_image.npy")
    test_mask = np.load(".//numpy_data/test_mask.npy")

    test_image_tensor = torch.from_numpy(test_image)
    test_mask_tensor = torch.from_numpy(test_mask)

    test_dataset = TensorDataset(test_image_tensor, test_mask_tensor)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=workers,
                             pin_memory=False,
                             prefetch_factor=1)

    return test_loader


def test_data_loader_ood(batch_size=1, workers=4):
    test_image_ood = np.load(".//numpy_data/test_image_ood.npy")
    test_mask_ood = np.load(".//numpy_data/test_mask.npy")

    test_image_tensor = torch.from_numpy(test_image_ood)
    test_mask_tensor = torch.from_numpy(test_mask_ood)

    print(test_image_tensor.shape)
    print(test_mask_tensor.shape)

    test_dataset = TensorDataset(test_image_tensor, test_mask_tensor)

    test_loader_ood = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=workers,
                                 pin_memory=False,
                                 prefetch_factor=1)
    return test_loader_ood


if __name__ == '__main__':
    print("data_loader")
    train_data_loader()