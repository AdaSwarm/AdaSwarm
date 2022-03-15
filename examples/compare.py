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


os.environ["USE_ADASWARM"] = "True"

metrics = main.run()

adaswarm_epoch_losses = metrics.epoch_train_losses


plt.figure(figsize=(20, 10))

plt.plot(adam_epoch_losses, label="Adam")
plt.plot(adaswarm_epoch_losses, label="AdaSwarm")

plt.title("Adam versus Adaswarm loss")

plt.legend()

plt.savefig(os.path.join("report", f"{options.dataset_name()}-training_loss.png"))

print(f"adam: {adam_epoch_losses}")
print(f"adaswarm: {adaswarm_epoch_losses}")