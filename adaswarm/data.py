from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
