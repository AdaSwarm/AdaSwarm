#!/usr/bin/env python3

import subprocess

import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
import pandas as pd

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, ".."))

import examples.main as main

import adaswarm.utils.options as options

os.environ["USE_ADASWARM"] = "False"


(
    train_losses_dataframe,
    train_accuracies_dataframe,
    test_losses_dataframe,
    test_accuracies_dataframe,
) = (
    pd.DataFrame(),
    pd.DataFrame(),
    pd.DataFrame(),
    pd.DataFrame(),
)

def write_run_to_dataframe(dataframe, dataset, name, run_number):
    this_run_dataframe = pd.DataFrame()
    this_run_dataframe["Epoch"] = range(options.number_of_epochs())
    this_run_dataframe["Name"] = name
    this_run_dataframe["Value"] = dataset
    this_run_dataframe["Run"] = run_number
    return pd.concat([dataframe, this_run_dataframe]).reset_index(drop=True)



compare_number_runs = 3
for run_number in range(compare_number_runs):

    metrics = main.run()
    (
        adam_epoch_train_losses,
        adam_epoch_train_accuracies,
        adam_epoch_test_losses,
        adam_epoch_test_accuracies,
    ) = metrics.run_data()

    train_losses_dataframe = write_run_to_dataframe(train_losses_dataframe, adam_epoch_train_losses, "Adam", run_number)
    train_accuracies_dataframe = write_run_to_dataframe(train_accuracies_dataframe, adam_epoch_train_accuracies, "Adam", run_number)
    test_losses_dataframe = write_run_to_dataframe(test_losses_dataframe, adam_epoch_test_losses, "Adam", run_number)
    test_accuracies_dataframe = write_run_to_dataframe(test_accuracies_dataframe, adam_epoch_test_accuracies, "Adam", run_number)


print(train_losses_dataframe.head())
print(train_accuracies_dataframe.head())
print(test_losses_dataframe.head())
print(test_accuracies_dataframe.head())

os.environ["USE_ADASWARM"] = "True"

metrics = main.run()
(
    adaswarm_epoch_train_losses,
    adaswarm_epoch_train_accuracies,
    adaswarm_epoch_test_losses,
    adaswarm_epoch_test_accuracies,
) = metrics.run_data()

def plot_aggregate(dataframe, title):
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(data=dataframe, x="Epoch", y="Value", hue="Name", estimator="mean", ax=ax)
    ax.set_title(title)
    filename = "-".join(title.lower().split(" "))
    fig.savefig(os.path.join("report", f"{options.dataset_name()}-aggregate-{filename}.png"))


def plot(adam_data, adaswarm_data, title):
    plt.figure(figsize=(20, 10))
    plt.plot(adam_data, label="Adam")
    plt.plot(adaswarm_data, label="AdaSwarm")
    plt.title(title)
    plt.legend()
    filename = "-".join(title.lower().split(" "))
    plt.savefig(os.path.join("report", f"{options.dataset_name()}-{filename}.png"))


# plot(
#     adam_data=adam_epoch_train_accuracies,
#     adaswarm_data=adaswarm_epoch_train_accuracies,
#     title="Training Accuracy",
# )

plot_aggregate(dataframe=train_accuracies_dataframe, title="Training Accuracy")

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
