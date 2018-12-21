"""Wrapper around mnist dataset"""
import torch
from torch.utils.data import Dataset

from torchvision.datasets.mnist import MNIST

# %%

class MyMNIST(Dataset):
    """MNIST dataset which can load all the images and labels on cpu or gpu.
    """
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 device='cpu'):

        dset = MNIST(root, train, transform, target_transform, download)
        if train:
            self.data = dset.train_data.to(device)
            self.labels = dset.train_labels.to(device)
        else:
            self.data = dset.test_data.to(device)
            self.labels = dset.test_labels.to(device)

        self.data = self.data.float()/255.0 - 0.5
        self.data = self.data.unsqueeze(1)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return self.data.shape[0]