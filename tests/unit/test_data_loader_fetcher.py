import unittest
from unittest.mock import patch, ANY

import numpy as np
from numpy.testing import assert_array_almost_equal

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import tensor
from adaswarm.model import Model
from sklearn import datasets

from adaswarm.data import DataLoaderFetcher, TabularDataSet
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
            TabularDataSet(train=True, dataset=datasets.load_iris)
            mock.assert_called()
        predictors, species = TabularDataSet(
            train=True, dataset=datasets.load_iris
        ).__getitem__([50])
        self.assertIsNone(
            assert_array_almost_equal(
                np.array(predictors[0]), np.array([6.4, 3.2, 4.5, 1.5])
            )
        )
        self.assertEqual(species[0][0], tensor([0]))

    def test_load_iris_training_set(self):
        fetcher = DataLoaderFetcher(name="Iris")
        with patch(
            "sklearn.datasets.load_iris", return_value=datasets.load_iris()
        ) as mock:
            loader = fetcher.train_loader()
            mock.assert_called()

        self.assertIsInstance(loader, DataLoader)

    def test_load_wine_training_set(self):
        fetcher = DataLoaderFetcher(name="Wine")
        with patch(
            "sklearn.datasets.load_wine", return_value=datasets.load_wine()
        ) as mock:
            loader = fetcher.train_loader()
            mock.assert_called()

        self.assertIsInstance(loader, DataLoader)

    def test_load_iris_test_set(self):
        TabularDataSet(train=False, dataset=datasets.load_iris)
        predictors, species = TabularDataSet(
            train=False, dataset=datasets.load_iris
        ).__getitem__([5])
        self.assertIsNone(
            assert_array_almost_equal(
                np.array(predictors[0]), np.array([4.7, 3.2, 1.3, 0.2])
            )
        )
        self.assertEqual(species[0][0], tensor([1]))

    def test_MNIST_model_selection(self):
        fetcher = DataLoaderFetcher(name="MNIST")
        chosen_model = fetcher.model()
        self.assertIsInstance(chosen_model, ResNet)

    def test_iris_model_selection(self):
        fetcher = DataLoaderFetcher(name="Iris")
        chosen_model = fetcher.model()
        self.assertIsInstance(chosen_model, Model)

    def test_wine_data(self):
        predictors, wine_types = TabularDataSet(
            train=True, dataset=datasets.load_wine
        ).__getitem__([50])
        self.assertIsNone(
            assert_array_almost_equal(
                np.array(predictors[0]),
                np.array(
                    [
                        13.56,
                        1.73,
                        2.46,
                        20.5,
                        116.0,
                        2.96,
                        2.78,
                        0.20,
                        2.45,
                        6.25,
                        0.98,
                        3.03,
                        1120,
                    ]
                ),
            )
        )
        self.assertEqual(wine_types[0][0], tensor([1]))

    def test_tabular_dataset_empty(self):
        with self.assertRaises(RuntimeError) as context:
            TabularDataSet()
        self.assertTrue("Dataset not provided" in str(context.exception))
