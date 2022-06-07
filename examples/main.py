#!/usr/bin/env python3

"""An example that uses AdaSwarm on MNIST dataset"""

import argparse
import logging
import os
import sys
import time

# pylint: disable=E0611
import torch
from torch.autograd.grad_mode import no_grad
from torch.autograd import Variable

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, ".."))


# pylint: disable=C0411, E0401, C0413
import adaswarm.nn
from adaswarm.utils import progress_bar
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


def run():
    chosen_loss_function = "AdaSwarm" if is_adaswarm() else "Adam"
    logging.debug("in run function")
    device = get_device()

    parser = argparse.ArgumentParser(
        description=f"PyTorch {dataset_name()} Training"
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
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

    print(f"Trainloader {len(trainloader.dataset)} dataset")
    print(f"Testloader {len(testloader.dataset)} dataset")
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

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    logging.info("Using %s Optimiser", chosen_loss_function)

    #TODO: Use a candidate loss function on a case by case basis
    if dataset_name() in ["Iris"]:
        if is_adaswarm():
            approx_criterion = adaswarm.nn.BCELoss()
        else:
            approx_criterion = torch.nn.BCELoss()
    else:
        if is_adaswarm():
            approx_criterion = adaswarm.nn.CrossEntropyLoss()
        else:
            approx_criterion = torch.nn.CrossEntropyLoss()

    # Training
    def train(epoch):
        print(f"\nEpoch: {epoch}")
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        batch_accuracies = []
        batch_losses = []

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tic = time.monotonic()
            if chosen_loss_function.lower() in ["adaswarm"]:
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
            )  # loss.item() contains the loss of entire mini-batch,
            # but divided by the batch size.
            # here we are summing up the losses as we go

            if dataset_name() in ["Iris"]:
                accuracy = (
                    torch.eq(outputs.round(), targets).float().mean().item()
                )  # accuracy
            else:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                accuracy = correct / total

            batch_accuracies.append(accuracy)
            toc = time.monotonic()
            batch_losses.append(loss.item())
            print(f"Batch : {batch_idx}| Loss: {loss.item()} | Time: {toc-tic}")


    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        running_loss = 0

        batch_accuracies = []
        batch_losses = []

        if dataset_name() in ["Iris"]:
            criterion = torch.nn.BCELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

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
                        torch.eq(outputs.round(), targets).float().mean().item()
                    )  # accuracy
                else:
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    accuracy = correct / total
                test_loss = running_loss / (batch_idx + 1)
                batch_accuracies.append(accuracy)
                batch_losses.append(loss.item())

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
# read the csv output to compare both runs
if __name__ == "__main__":
    run()
