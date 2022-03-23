import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader


def numpy_data_loader():
    train_flair = np.load(".//data/train_flair.npy")
    train_mask = np.load(".//data/train_mask.npy")
    val_flair = np.load(".//data/val_flair.npy")
    val_mask = np.load(".//data/val_mask.npy")
    test_flair = np.load(".//data/test_flair.npy")
    test_mask = np.load(".//data/test_mask.npy")

    return train_flair, train_mask, val_flair, val_mask, test_flair, test_mask


def torch_data_loader(batch_size=2, workers=4):
    train_flair, train_mask, val_flair, val_mask, test_flair, test_mask = numpy_data_loader()

    train_image_tensor = torch.from_numpy(train_flair)
    train_mask_tensor = torch.from_numpy(train_mask)

    valid_image_tensor = torch.from_numpy(val_flair)
    valid_mask_tensor = torch.from_numpy(val_mask)

    test_image_tensor = torch.from_numpy(test_flair)
    test_mask_tensor = torch.from_numpy(test_mask)

    train_dataset = TensorDataset(train_image_tensor, train_mask_tensor)
    valid_dataset = TensorDataset(valid_image_tensor, valid_mask_tensor)
    test_dataset = TensorDataset(test_image_tensor, test_mask_tensor)

    print(train_image_tensor.shape)
    print(test_mask_tensor.shape)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers,
                              pin_memory=True,
                              prefetch_factor=1

                              )

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=workers,
                              pin_memory=False,
                              prefetch_factor=1)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=workers,
                             pin_memory=False,
                             prefetch_factor=1)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    print("data_loader")
    # numpy_data_loader()
    torch_data_loader()
