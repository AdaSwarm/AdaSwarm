from torch.nn.parallel import DataParallel, distributed
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import is_tensor, from_numpy
import numpy as np
from torchvision import transforms
from torch import cuda
from torch.backends import cudnn

from adaswarm.utils.options import get_device
from adaswarm.resnet import ResNet18

from sklearn import datasets as skl_datasets
from torchvision import datasets as tv_datasets
from sklearn.model_selection import train_test_split

device = get_device()


class DataLoaderFetcher:
    def __init__(self, name: str = "MNIST"):
        self.name = name

    # TODO: Handle error in case user passes an unsupported dataset name
    def train_loader(self):

        if self.name == "Iris":
            return DataLoader(
                IrisDataSet(),
                batch_size=2,
                shuffle=True,
                drop_last=False,
            )

        else:
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
                tv_datasets.MNIST(
                    root="./data", train=True, download=True, transform=transform_train
                ),
                batch_size=125,
                shuffle=True,
                num_workers=2,
            )

    def test_loader(self):
        if self.name == "Iris":
            return DataLoader(
                IrisDataSet(train=False),
                batch_size=2,
                shuffle=True,
                drop_last=False,
            )
        else:
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            return DataLoader(
                tv_datasets.MNIST(
                    root="./data", train=False, download=True, transform=transform_test
                ),
                batch_size=100,
                shuffle=False,
                num_workers=2,
            )

    def model(self):
        model = ResNet18(in_channels=1, num_classes=10)
        model = model.to(device)
        if cuda.is_available():
            cudnn.benchmark = True
            # TODO: Use torch.nn.parallel.DistributedDataParallel
            model = DataParallel(model)
        return model


class IrisDataSet(Dataset):
    def __init__(self, train=True):
        iris_data_bundle = skl_datasets.load_iris()
        x, y = iris_data_bundle.data, iris_data_bundle.target
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=123
        )
        if train:
            self.data = x_train
            self.target = y_train
        else:
            self.data = x_test
            self.target = y_test

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        predictors = from_numpy(self.data[idx, 0:4]).to(device)
        species = from_numpy(np.array(self.target[idx])).to(device)
        return predictors, species

    def __len__(self):
        return len(self.data)
