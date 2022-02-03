from sys import platform
import os
import multiprocessing
import torch
from torch.nn import functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from adaswarm.resnet import ResNet18

if platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,), (0.2023,)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914,), (0.2023,)),
    ]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform_train
)
trainloader = DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
    num_workers=multiprocessing.cpu_count(),
)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)


class LitResnet18Adam(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat = self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.1)


if __name__ == "__main__":
    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = ResNet18(1)
    net = net.to(device)
    if device == "cuda":
        net = DataParallel(net)
        cudnn.benchmark = True

    # init model
    resnet = LitResnet18Adam(net)

    # Initialize a trainer
    trainer = pl.Trainer(gpus=0, max_epochs=200)

    # Train the model
    trainer.fit(resnet, trainloader, testloader)
