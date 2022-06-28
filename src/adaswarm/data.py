"""Data to supply the AdaSwarm example - using Iris dataset"""

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import is_tensor

from sklearn import datasets as skl_datasets
from sklearn.model_selection import StratifiedShuffleSplit

from adaswarm.utils import to_categorical
from adaswarm.utils.options import get_device
from adaswarm.model import Model

device = get_device()


class DataLoaderFetcher:
    """Data loaders for Iris dataset"""

    # TODO: using for now and will be removed in future once Taimatsu is ready
    def __init__(self, name: str = "Iris"):
        self.name = name

    # TODO: Handle error in case user passes an unsupported dataset name
    def train_loader(self) -> DataLoader:
        """Train loader for Iris dataset"""

        # TODO: make Iris the default

        return DataLoader(
            self.dataset(train=True),
            batch_size=40,
            shuffle=True,
            drop_last=False,
        )

    def test_loader(self) -> DataLoader:
        """Test loader for Iris dataset"""
        return DataLoader(
            self.dataset(train=False),
            batch_size=10,
            shuffle=True,
            drop_last=False,
        )

    def dataset(self, train=True):  # pylint: disable=R0201
        """Iris dataset"""
        return TabularDataSet(train=train, dataset=skl_datasets.load_iris)

    def model(self):
        """Model suitable for Iris dataset"""
        return Model(
            n_features=self.dataset().number_of_predictors(),
            n_neurons=10,
            n_out=self.dataset().number_of_categories(),
        )


class TabularDataSet(Dataset):
    """Tabular dataset constructor"""

    def __init__(self, train=True, dataset=None):
        if dataset is None:
            raise RuntimeError("Dataset not provided")

        data_bundle = dataset()

        x, y = data_bundle.data, data_bundle.target  # pylint: disable=C0103
        y_categorical = to_categorical(y)

        self._number_of_predictors = np.shape(x)[1]
        self._number_of_categories = np.shape(y_categorical)[1]

        stratified_shuffle_split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=123
        )

        for train_index, test_index in stratified_shuffle_split.split(X=x, y=y):
            x_train_array = x[train_index]
            x_test_array = x[test_index]
            y_train_array = y_categorical[train_index]
            y_test_array = y_categorical[test_index]

        if train:

            self.data = x_train_array
            self.target = y_train_array
        else:

            self.data = x_test_array
            self.target = y_test_array

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        # TODO: may be repetition of from_numpy here
        predictors = torch.from_numpy(self.data[idx, :]).float().to(device)
        categories = torch.from_numpy(self.target[idx]).to(device)
        return predictors, categories

    def number_of_predictors(self):
        """Get number of predictors"""
        return self._number_of_predictors

    def number_of_categories(self):
        """Get number of categories"""
        return self._number_of_categories

    def __len__(self):
        return len(self.data)
