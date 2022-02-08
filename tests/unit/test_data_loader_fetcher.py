from typing import Any
import unittest
import os
from unittest.mock import patch, ANY

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from adaswarm.utils.options import dataset_name
from adaswarm.data import DataLoaderFetcher


class CustomDataset(Dataset):
    def __init__(self):
        self.data = [1]

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


class DataLoaderTestCase(unittest.TestCase):
    def test_load_MNIST_Train_set(self):
        os.environ["ADASWARM_DATASET_NAME"] = "MNIST"
        fetcher = DataLoaderFetcher(dataset_name())
        with patch(
            "torchvision.datasets.MNIST", return_value=DataLoader(CustomDataset())
        ) as mock:
            loader = fetcher.train_loader()
            mock.assert_called_with(
                root="./data", train=True, download=True, transform=ANY
            )

        self.assertIsInstance(loader, DataLoader)

    def test_load_MNIST_test_set(self):
        os.environ["ADASWARM_DATASET_NAME"] = "MNIST"
        fetcher = DataLoaderFetcher(dataset_name())
        with patch(
            "torchvision.datasets.MNIST", return_value=DataLoader(CustomDataset())
        ) as mock:
            loader = fetcher.test_loader()
            mock.assert_called_with(
                root="./data", train=False, download=True, transform=ANY
            )
        self.assertIsInstance(loader, DataLoader)
