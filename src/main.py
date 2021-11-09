#!/usr/bin/env python3

"""Train MNIST with PyTorch."""
# pylint: disable=C0411
from nn_utils import CELossWithPSO
from utils import progress_bar
from resnet import ResNet18
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.backends import cudnn

# pylint: disable=E0611
from torch import (
    nn,
    no_grad,
    optim,
    cuda,
    load as torch_load,
    save as torch_save,
    device as torch_device,
)
from torchvision import transforms, datasets
import os
import argparse
import logging

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)

# pylint: disable=R0914,R0915,C0116,C0413


def run():
    print("in run function")
    device = torch_device("cuda:0" if cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="PyTorch MNIST Training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )

    args = parser.parse_args()

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
    approx_criterion = CELossWithPSO.apply

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
            print("PSO ran...")
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = approx_criterion(
                outputs,
                targets,
                0.4,
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print_output = f"""Loss: {train_loss/(batch_idx+1):3f}
                    | Acc: {100.*correct/total}%% ({correct/total})"""

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
