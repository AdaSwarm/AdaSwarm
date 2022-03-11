#!/usr/bin/env python3

"""An example that uses AdaSwarm on MNIST dataset"""

import argparse
import logging
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

# pylint: disable=E0611
import torch
from torch.autograd.grad_mode import no_grad
from torch.autograd import Variable

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, ".."))


# pylint: disable=C0411, E0401, C0413
import adaswarm.nn
from adaswarm.utils import progress_bar, Metrics
from adaswarm.data import DataLoaderFetcher

from adaswarm.utils.options import (
    is_adaswarm,
    number_of_epochs,
    dataset_name,
    get_device,
    log_level,
)


logging.basicConfig(level=log_level())

# pylint: disable=R0914,R0915,C0116,C0413

CHOSEN_LOSS_FUNCTION = "AdaSwarm" if is_adaswarm() else "Adam"

epoch_train_losses = []
epoch_train_accuracies = []


def run():
    logging.debug("in run function")
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

    num_batches_train = int(len(trainloader.dataset) / trainloader.batch_size)

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
        if dataset_name() in ["Iris"]:
            approx_criterion = adaswarm.nn.BCELoss()
        else:
            approx_criterion = adaswarm.nn.CrossEntropyLoss()
    else:
        if dataset_name() in ["Iris"]:
            approx_criterion = torch.nn.BCELoss()
        else:
            approx_criterion = torch.nn.CrossEntropyLoss()

    # Training
    def train(epoch):
        print(f"\nEpoch: {epoch}")
        metrics.current_epoch(epoch + 1)
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        batch_accuracies = []
        batch_losses = []

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets.requires_grad = False

            if dataset_name() in ["Iris"]:
                inputs = Variable(inputs).float()
                targets = Variable(
                    targets,
                ).float()
                outputs = model(inputs).float()
            else:
                outputs = model(inputs)

            loss = approx_criterion(outputs, targets)

            optimizer.zero_grad()  # zero the gradients on each pass before the update
            loss.backward()  # backpropagate the loss through the model
            optimizer.step()  # update the gradients w.r.t the loss

            running_loss += (
                loss.item()
            )  # loss.item() contains the loss of entire mini-batch, but divided by the batch size.
            # here we are summing up the losses as we go
            batch_losses.append(loss.item())
            print(f"Batch : {batch_idx}| Loss: {loss.item()}")

            if dataset_name() in ["Iris"]:
                accuracy = (
                    torch.eq(outputs.round(), targets).float().mean().item()
                )  # accuracy
            else:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                accuracy = correct / total

            training_loss = running_loss / (batch_idx + 1)

            batch_accuracies.append(accuracy)
            metrics.update_training_accuracy(accuracy)
            metrics.update_training_loss(training_loss)

        epoch_train_losses.append(sum(batch_losses) / num_batches_train)
        epoch_train_accuracies.append(100 * sum(batch_accuracies) / num_batches_train)

        if epoch % 1 == 0:
            print(
                f"[{epoch}/{number_of_epochs()}], \
                loss: {np.round(sum(batch_losses) / num_batches_train, 3)} \
                    acc: {100 * np.round(sum(batch_accuracies) / num_batches_train, 3)}"
            )

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        running_loss = 0

        with no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                if dataset_name() in ["Iris"]:
                    inputs = Variable(inputs).float()
                    targets = Variable(
                        targets,
                    ).float()
                    outputs = model(inputs).float()
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()

                if dataset_name() in ["Iris"]:
                    accuracy = (
                        torch.eq(outputs.round(), targets).float().mean()
                    )  # accuracy
                else:
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

        # Save checkpoint.
        acc = 100.0 * accuracy

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
        plt.figure(figsize=(20, 10))
        plt.title(dataset_name() + " Loss")
        plt.plot(epoch_train_losses, label=CHOSEN_LOSS_FUNCTION)
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 10))
        plt.title(dataset_name() + " Accuracy")
        plt.plot(epoch_train_accuracies, label=CHOSEN_LOSS_FUNCTION)
        plt.legend()
        plt.show()
