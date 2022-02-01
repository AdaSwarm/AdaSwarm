"""
    A module for reporting metrics for training and
    testing runs
"""
import os
import time
import datetime
import pandas as pd
import numpy as np


class Metrics:
    """
    Metrics class to capture
    * Time taken to run
    * Accuracy
    """

    class Stats:
        """
        Class to hold the value of accuracy
        """

        def __init__(self):
            self.best_training_accuracy = 0.0
            self.best_training_loss = None
            self.best_test_loss = None
            self.number_of_epochs = 0

        def update_training_accuracy(self, value):
            """
            Compare and store the best accuracy
            value
            """

            # TODO: Store every epoch and accuracy
            # TODO: Take epoch as an argument and store the best accuracy
            # and the epoch when it was achieved
            if value > self.best_training_accuracy:
                self.best_training_accuracy = value

        def update_training_loss(self, value):
            """
            Compare and store the best training loss
            value
            """
            if (self.best_training_loss == None) or (value < self.best_training_loss):
                self.best_training_loss = value

        def update_test_loss(self, value):
            """
            Compare and store the best test loss
            value
            """
            if (self.best_test_loss == None) or (value < self.best_test_loss):
                self.best_test_loss = value

        def current_epoch(self, value):
            self.number_of_epochs = value

    def __init__(
        self,
        name: str = "Default run",
        md_filepath: str = os.path.join("report", "summary.md"),
        csv_filepath: str = os.path.join("report", "summary.csv"),
    ):
        self.md_filepath = md_filepath
        self.csv_filepath = csv_filepath
        self.name = name
        self.stats = Metrics.Stats()
        self.name = name
        self.tstart = time.time()

    def __enter__(self):
        return self.stats

    def __exit__(self, *args):

        time_taken = time.time() - self.tstart
        this_summary_dataframe = pd.DataFrame(
            {
                "Start time": datetime.datetime.fromtimestamp(self.tstart).strftime(
                    "%d-%m-%y %H:%M:%S"
                ),
                "Run name": self.name,
                "Number of epochs": self.stats.number_of_epochs,
                "Elapsed (seconds)": time_taken,
                "Training Accuracy %": np.round(
                    100.0 * self.stats.best_training_accuracy, 2
                ),
                "Training Loss": self.stats.best_training_loss
                if self.stats.best_training_loss != None
                else "Not set",
                "Test Loss": self.stats.best_test_loss
                if self.stats.best_test_loss != None
                else "Not set",
            },
            index=[0],
        )
        os.makedirs(os.path.dirname(self.md_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_filepath), exist_ok=True)

        if os.path.exists(self.csv_filepath):
            previous_summary_dataframe = pd.read_csv(self.csv_filepath, index_col=None)
        else:
            previous_summary_dataframe = pd.DataFrame()

        this_summary_dataframe = pd.concat(
            [previous_summary_dataframe, this_summary_dataframe], axis=0
        )

        with open(self.md_filepath, mode="w", encoding="utf-8") as file:
            this_summary_dataframe.to_markdown(buf=file, index=False)

        this_summary_dataframe.to_csv(self.csv_filepath, index=False)
