import unittest
from unittest.mock import patch, ANY

import numpy as np
from numpy.testing import assert_array_almost_equal

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import tensor
from adaswarm.model import Model
from sklearn import datasets

from adaswarm.data import DataLoaderFetcher, IrisDataSet
from adaswarm.resnet import ResNet


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
        with patch("torchvision.datasets.MNIST", return_value=CustomDataset()) as mock:
            loader = fetcher.train_loader()
            mock.assert_called_with(
                root="./data", train=True, download=True, transform=ANY
            )

        self.assertIsInstance(loader, DataLoader)

    def test_load_MNIST_test_set(self):
        fetcher = DataLoaderFetcher(name="MNIST")
        with patch("torchvision.datasets.MNIST", return_value=CustomDataset()) as mock:
            loader = fetcher.test_loader()
            mock.assert_called_with(
                root="./data", train=False, download=True, transform=ANY
            )
        self.assertIsInstance(loader, DataLoader)

    def test_iris_data(self):
        with patch(
            "sklearn.datasets.load_iris", return_value=datasets.load_iris()
        ) as mock:
            IrisDataSet()
            mock.assert_called()
        predictors, species = IrisDataSet().__getitem__([50])
        self.assertIsNone(
            assert_array_almost_equal(
                np.array(predictors[0]), np.array([5.0, 3.5, 1.6, 0.6])
            )
        )
        self.assertEqual(species[0][0], tensor([1]))

    def test_load_iris_training_set(self):
        fetcher = DataLoaderFetcher(name="Iris")
        with patch(
            "sklearn.datasets.load_iris", return_value=datasets.load_iris()
        ) as mock:
            loader = fetcher.train_loader()
            mock.assert_called()

        self.assertIsInstance(loader, DataLoader)

    def test_load_iris_test_set(self):
        with patch(
            "sklearn.datasets.load_iris", return_value=datasets.load_iris()
        ) as mock:
            IrisDataSet(train=False)
            mock.assert_called()
        predictors, species = IrisDataSet(train=False).__getitem__([5])
        self.assertIsNone(
            assert_array_almost_equal(
                np.array(predictors[0]), np.array([6.0, 3.0, 4.8, 1.8])
            )
        )
        self.assertEqual(species[0][0], tensor([0]))

    def test_MNIST_model_selection(self):
        fetcher = DataLoaderFetcher(name="MNIST")
        chosen_model = fetcher.model()
        self.assertIsInstance(chosen_model, ResNet)

    def test_iris_model_selection(self):
        fetcher = DataLoaderFetcher(name="Iris")
        chosen_model = fetcher.model()
        self.assertIsInstance(chosen_model, Model)
