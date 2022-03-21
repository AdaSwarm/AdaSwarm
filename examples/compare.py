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

adam_epoch_losses = metrics.epoch_train_losses
adam_epoch_accuracies = metrics.epoch_train_accuracies


os.environ["USE_ADASWARM"] = "True"

metrics = main.run()

adaswarm_epoch_losses = metrics.epoch_train_losses
adaswarm_epoch_accuracies = metrics.epoch_train_accuracies


plt.figure(figsize=(20, 10))

plt.plot(adam_epoch_losses, label="Adam")
plt.plot(adaswarm_epoch_losses, label="AdaSwarm")

plt.title("Adam versus Adaswarm loss")

plt.legend()

plt.savefig(os.path.join("report", f"{options.dataset_name()}-training_loss.png"))


plt.figure(figsize=(20, 10))

plt.plot(adam_epoch_accuracies, label="Adam")
plt.plot(adaswarm_epoch_accuracies, label="AdaSwarm")

plt.title("Adam versus Adaswarm accuracy")

plt.legend()

plt.savefig(os.path.join("report", f"{options.dataset_name()}-training_accuracy.png"))
