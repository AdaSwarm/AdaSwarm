#!/usr/bin/env python3

"""An example that uses AdaSwarm on MNIST dataset"""

import argparse
import logging
import os
import sys

# pylint: disable=E0611
from torch import cuda
from torch import device as torch_device
from torch import load as torch_load
from torch import nn, optim
from torch.autograd.grad_mode import no_grad
from torch import save as torch_save
from torch.backends import cudnn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, ".."))


# pylint: disable=C0411, E0401, C0413
from adaswarm.nn_utils import CELossWithPSO
from adaswarm.resnet import ResNet18
from adaswarm.utils import progress_bar
from adaswarm.utils.options import is_adaswarm

# Writer will output to ./runs/ directory by default
TENSORBOARD_LOG_DIR = "runs/adaswarm" if is_adaswarm() else "runs/adam"
writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)

# pylint: disable=R0914,R0915,C0116,C0413


def run():
    print("in run function")
    device = torch_device("cuda:0" if cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="PyTorch MNIST Training")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )

    args = parser.parse_args()

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            # Image Transformations suitable for MNIST dataset(handwritten digits)
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            # Mean and Std deviation values of MNIST dataset
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform_train
    )

    trainloader = DataLoader(trainset, batch_size=125, shuffle=True, num_workers=2)

    testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    print("==> Building model..")
    net = ResNet18(1)
    net = net.to(device)
    if device == "cuda":
        net = DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch_load("./checkpoint/ckpt.pth")
        net.load_state_dict(checkpoint["net"])
        start_epoch = checkpoint["epoch"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if is_adaswarm():
        logging.info("Using Swarm Optimiser")
        approx_criterion = CELossWithPSO.apply
        chosen_optimizer = "Adaswarm"
    else:
        logging.info("Using Adam Optimiser")
        approx_criterion = nn.CrossEntropyLoss()
        chosen_optimizer = "Adam"

    # Training
    def train(epoch):
        print(f"\nEpoch: {epoch}")
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            logging.debug("targets: %s", targets)
            targets.requires_grad = False
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = approx_criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print_output = f"""Loss: {train_loss/(batch_idx+1):3f}
                    | Acc: {100.*correct/total}%% ({correct/total})"""

            if batch_idx % 50 == 49:  # every 50 mini-batches...

                writer.add_scalar(
                    tag=f"{chosen_optimizer}/train loss",
                    scalar_value=train_loss / (batch_idx + 1),
                    global_step=epoch * len(trainloader) + batch_idx + 1,
                )
                writer.add_scalar(
                    tag=f"{chosen_optimizer}/train accuracy",
                    scalar_value=correct / total,
                    global_step=epoch * len(trainloader) + batch_idx + 1,
                )

            print(batch_idx, len(trainloader), print_output)
            progress_bar(batch_idx, len(trainloader), print_output)

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(
                    batch_idx,
                    len(testloader),
                    f"""Loss: {test_loss/(batch_idx+1):3f}
                    | Acc: {100.*correct/total}%% ({correct/total})""",
                )
                if batch_idx % 50 == 49:  # every 50 mini-batches...

                    writer.add_scalar(
                        tag=f"{chosen_optimizer}/test loss",
                        scalar_value=test_loss / (batch_idx + 1),
                        global_step=epoch * len(testloader) + batch_idx + 1,
                    )
                    writer.add_scalar(
                        tag=f"{chosen_optimizer}/test accuracy",
                        scalar_value=correct / total,
                        global_step=epoch * len(testloader) + batch_idx + 1,
                    )

        # Save checkpoint.
        acc = 100.0 * correct / total

        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch_save(state, "./checkpoint/ckpt.pth")

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)


if __name__ == "__main__":
    run()
