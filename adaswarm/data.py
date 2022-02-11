from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import is_tensor, from_numpy
from torchvision import datasets, transforms

from adaswarm.utils.options import get_device

from sklearn.datasets import load_iris
import sklearn

device = get_device()


class DataLoaderFetcher:
    def __init__(self, name: str):
        self.name = name

    def train_loader(self):
        if self.name == "MNIST":
            transform_train = transforms.Compose(
                [
                    # Image Transformations suitable for MNIST dataset(handwritten digits)
                    transforms.RandomRotation(30),
                    transforms.RandomAffine(
                        degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    # Mean and Std deviation values of MNIST dataset
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            return DataLoader(
                datasets.MNIST(
                    root="./data", train=True, download=True, transform=transform_train
                ),
                batch_size=125,
                shuffle=True,
                num_workers=2,
            )

        elif self.name == "Iris":
            return DataLoader(
                IrisDataSet(),
                batch_size=2,
                shuffle=True,
                drop_last=False,
            )


    def test_loader(self):
        if self.name == "MNIST":
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            return DataLoader(
                datasets.MNIST(
                    root="./data", train=False, download=True, transform=transform_test
                ),
                batch_size=100,
                shuffle=False,
                num_workers=2,
            )


class IrisDataSet(Dataset):
    def __init__(self):
        iris_data_bundle = sklearn.datasets.load_iris()
        self.data = iris_data_bundle.data
        self.target = iris_data_bundle.target

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        predictors = from_numpy(self.data[idx, 0:4]).to(device)
        species = from_numpy(self.target[idx]).to(device)
        sample = {"predictors": predictors, "species": species}
        return sample

    def __len__(self):
        return len(self.data)
