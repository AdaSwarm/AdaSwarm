#!/usr/bin/env python3

"""An example that uses AdaSwarm on MNIST dataset"""

import argparse
import logging
import os
import sys

# pylint: disable=E0611
import torch
from torch.autograd.grad_mode import no_grad
from torch.utils.tensorboard.writer import SummaryWriter

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, ".."))


# pylint: disable=C0411, E0401, C0413
import adaswarm.nn
from adaswarm.utils import progress_bar, Metrics
from adaswarm.data import DataLoaderFetcher

from adaswarm.utils.options import (
    is_adaswarm,
    get_tensorboard_log_path,
    number_of_epochs,
    write_to_tensorboard,
    dataset_name,
    get_device,
    log_level,
)

# TODO: allow running without tensorboard option
writer_1 = SummaryWriter(get_tensorboard_log_path("train"))
writer_2 = SummaryWriter(get_tensorboard_log_path("eval"))

logging.basicConfig(level=log_level())

# pylint: disable=R0914,R0915,C0116,C0413

CHOSEN_LOSS_FUNCTION = "AdaSwarm" if is_adaswarm() else "Adam"


def run():
    print("in run function")
    device = get_device()

    parser = argparse.ArgumentParser(description=f"PyTorch {dataset_name()} Training")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )

    args = parser.parse_args()

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print("==> Preparing data..")

    fetcher = DataLoaderFetcher(dataset_name())
    trainloader = fetcher.train_loader()
    testloader = fetcher.test_loader()

    # Model
    print("==> Building model..")
    model = fetcher.model()

    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load("./checkpoint/ckpt.pth")
        model.load_state_dict(checkpoint["net"])
        start_epoch = checkpoint["epoch"]

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logging.info("Using %s Optimiser", CHOSEN_LOSS_FUNCTION)

    if is_adaswarm():
        approx_criterion = adaswarm.nn.CrossEntropyLoss.apply
    else:
        torch.nn.CrossEntropyLoss()

    # Training
    def train(epoch):
        print(f"\nEpoch: {epoch}")
        metrics.current_epoch(epoch + 1)
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets.requires_grad = False
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = approx_criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            accuracy = correct / total
            training_loss = running_loss / (batch_idx + 1)

            metrics.update_training_accuracy(accuracy)
            metrics.update_training_loss(training_loss)

            print_output = f"""Loss: {training_loss:3f}
                    | Acc: {100.*accuracy}%% ({accuracy})"""

            if write_to_tensorboard(batch_idx):  # every X mini-batches...

                writer_1.add_scalar(
                    tag="ada_vs_adam/train loss",
                    scalar_value=training_loss,
                    global_step=epoch * len(trainloader) + batch_idx + 1,
                )
                writer_1.add_scalar(
                    tag="ada_vs_adam/train accuracy",
                    scalar_value=accuracy,
                    global_step=epoch * len(trainloader) + batch_idx + 1,
                )

            print(batch_idx, len(trainloader), print_output)
            progress_bar(batch_idx, len(trainloader), print_output)

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        running_loss = 0

        with no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                accuracy = correct / total
                test_loss = running_loss / (batch_idx + 1)

                metrics.update_test_accuracy(accuracy)
                metrics.update_test_loss(test_loss)

                progress_bar(
                    batch_idx,
                    len(testloader),
                    f"""Loss: {test_loss:3f}
                    | Acc: {100.*accuracy}%% ({accuracy})""",
                )
                if write_to_tensorboard(batch_idx):  # every X mini-batches...

                    writer_2.add_scalar(
                        tag="ada_vs_adam/test loss",
                        scalar_value=test_loss,
                        global_step=epoch * len(testloader) + batch_idx + 1,
                    )
                    writer_2.add_scalar(
                        tag="ada_vs_adam/test accuracy",
                        scalar_value=accuracy,
                        global_step=epoch * len(testloader) + batch_idx + 1,
                    )

        # Save checkpoint.
        acc = 100.0 * correct / total

        print("Saving..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/ckpt.pth")

    for epoch in range(start_epoch, number_of_epochs()):
        train(epoch)
        test(epoch)


# TODO: Add a script to run both optimizers and
# read the csv ouput to compare both runs
if __name__ == "__main__":
    with Metrics(name=CHOSEN_LOSS_FUNCTION, dataset=dataset_name()) as metrics:
        run()
