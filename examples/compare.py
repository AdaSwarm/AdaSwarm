#!/usr/bin/env python3

import subprocess

import matplotlib.pyplot as plt

import sys
import os

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, ".."))

import examples.main as main

import adaswarm.utils.options as options

os.environ["USE_ADASWARM"] = "False"

metrics = main.run()
(
    adam_epoch_train_losses,
    adam_epoch_train_accuracies,
    adam_epoch_test_losses,
    adam_epoch_test_accuracies,
) = metrics.run_data()

os.environ["USE_ADASWARM"] = "True"

metrics = main.run()
(
    adaswarm_epoch_train_losses,
    adaswarm_epoch_train_accuracies,
    adaswarm_epoch_test_losses,
    adaswarm_epoch_test_accuracies,
) = metrics.run_data()


def plot(adam_data, adaswarm_data, title):
    plt.figure(figsize=(20, 10))
    plt.plot(adam_data, label="Adam")
    plt.plot(adaswarm_data, label="AdaSwarm")
    plt.title(title)
    plt.legend()
    filename = "-".join(title.lower().split(" "))
    plt.savefig(os.path.join("report", f"{options.dataset_name()}-{filename}.png"))


plot(
    adam_data=adam_epoch_train_accuracies,
    adaswarm_data=adaswarm_epoch_train_accuracies,
    title="Training Accuracy",
)

plot(
    adam_data=adam_epoch_train_losses,
    adaswarm_data=adaswarm_epoch_train_losses,
    title="Training Loss",
)

plot(
    adam_data=adam_epoch_test_accuracies,
    adaswarm_data=adaswarm_epoch_test_accuracies,
    title="Test Accuracy",
)

plot(
    adam_data=adam_epoch_test_losses,
    adaswarm_data=adaswarm_epoch_test_losses,
    title="Test loss",
)
