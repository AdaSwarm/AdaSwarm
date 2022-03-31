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


compare_number_runs = int(os.environ.get("ADASWARM_NUMBER_OF_RUNS", "5"))
for run_number in range(compare_number_runs):

    os.environ["USE_ADASWARM"] = "False"

    metrics = main.run()
    (
        adam_epoch_train_losses,
        adam_epoch_train_accuracies,
        adam_epoch_test_losses,
        adam_epoch_test_accuracies,
    ) = metrics.run_data()

    train_losses_dataframe = write_run_to_dataframe(
        train_losses_dataframe, adam_epoch_train_losses, "Adam", run_number
    )
    train_accuracies_dataframe = write_run_to_dataframe(
        train_accuracies_dataframe, adam_epoch_train_accuracies, "Adam", run_number
    )
    test_losses_dataframe = write_run_to_dataframe(
        test_losses_dataframe, adam_epoch_test_losses, "Adam", run_number
    )
    test_accuracies_dataframe = write_run_to_dataframe(
        test_accuracies_dataframe, adam_epoch_test_accuracies, "Adam", run_number
    )

    os.environ["USE_ADASWARM"] = "True"

    metrics = main.run()
    (
        adaswarm_epoch_train_losses,
        adaswarm_epoch_train_accuracies,
        adaswarm_epoch_test_losses,
        adaswarm_epoch_test_accuracies,
    ) = metrics.run_data()

    train_losses_dataframe = write_run_to_dataframe(
        train_losses_dataframe, adaswarm_epoch_train_losses, "Adaswarm", run_number
    )
    train_accuracies_dataframe = write_run_to_dataframe(
        train_accuracies_dataframe,
        adaswarm_epoch_train_accuracies,
        "Adaswarm",
        run_number,
    )
    test_losses_dataframe = write_run_to_dataframe(
        test_losses_dataframe, adaswarm_epoch_test_losses, "Adaswarm", run_number
    )
    test_accuracies_dataframe = write_run_to_dataframe(
        test_accuracies_dataframe,
        adaswarm_epoch_test_accuracies,
        "Adaswarm",
        run_number,
    )


def plot_aggregate(dataframe, title):
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(
        data=dataframe, x="Epoch", y="Value", hue="Name", estimator="mean", ax=ax
    )
    ax.set_title(f"Aggregated {title.lower()} over {compare_number_runs} runs")
    filename = "-".join(title.lower().split(" "))
    fig.savefig(
        os.path.join("report", f"{options.dataset_name()}-aggregate-{filename}.png")
    )


plot_aggregate(dataframe=train_accuracies_dataframe, title="Training Accuracy")
plot_aggregate(dataframe=train_losses_dataframe, title="Training Losses")
plot_aggregate(dataframe=test_accuracies_dataframe, title="Test Accuracy")
plot_aggregate(dataframe=test_losses_dataframe, title="Test Losses")
