from typing import Any
import unittest
import os
from unittest.mock import patch, ANY

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import tensor, float64

from adaswarm.utils.options import dataset_name
from adaswarm.data import DataLoaderFetcher, IrisDataSet


class CustomDataset(Dataset):
    def __init__(self):
        self.data = [1]

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


class DataLoaderTestCase(unittest.TestCase):
    def test_load_MNIST_Train_set(self):
        fetcher = DataLoaderFetcher(name="MNIST")
        with patch(
            "torchvision.datasets.MNIST", return_value=DataLoader(CustomDataset())
        ) as mock:
            loader = fetcher.train_loader()
            mock.assert_called_with(
                root="./data", train=True, download=True, transform=ANY
            )

        self.assertIsInstance(loader, DataLoader)

    def test_load_MNIST_test_set(self):
        fetcher = DataLoaderFetcher(name="MNIST")
        with patch(
            "torchvision.datasets.MNIST", return_value=CustomDataset()
        ) as mock:
            loader = fetcher.test_loader()
            mock.assert_called_with(
                root="./data", train=False, download=True, transform=ANY
            )
        self.assertIsInstance(loader, DataLoader)

    def test_load_iris_training_set(self):
        fetcher = DataLoaderFetcher(name="Iris")
        with patch(
            "sklearn.datasets.load_iris", return_value=IrisDataSet()
        ) as mock:
            loader = fetcher.train_loader()
            mock.assert_called()

        self.assertIsInstance(loader, DataLoader)

    def test_iris_data(self):
        with patch("sklearn.datasets.load_iris") as mock:
            IrisDataSet()
            mock.assert_called()
        iris_test_values = IrisDataSet().__getitem__([50])
        self.assertEqual(iris_test_values['predictors'].tolist()[0], [7.0000, 3.2000, 4.7000, 1.4000])
        self.assertEqual(iris_test_values['species'], tensor([1]))
