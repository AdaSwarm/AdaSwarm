"""
    A module for reporting metrics for training and
    testing runs
"""
import os
import time
import datetime
import pandas as pd
import numpy as np

OUTPUT_FILENAME = "./tests/unit/output/summary.md"


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
            self.best_accuracy = 0.0

        def update_train_accuracy(self, value):
            """
            Compare and store the best accuracy
            value
            """
            if value > self.best_accuracy:
                self.best_accuracy = value

    def __init__(
        self,
        name: str = "Default run",
        md_filepath: str = "./report/summary.md",
        csv_filepath: str = "./report/summary.csv",
    ):
        self.md_filepath = md_filepath
        self.csv_filepath = csv_filepath
        self.name = name
        self.accuracy = Metrics.Stats()
        self.name = name
        self.tstart = time.time()

    def __enter__(self):
        return self.accuracy

    def __exit__(self, *args):

        time_taken = time.time() - self.tstart
        this_summary_dataframe = pd.DataFrame(
            {
                "Start time": datetime.datetime.fromtimestamp(self.tstart).strftime(
                    "%d-%m-%y %H:%M:%S"
                ),
                "Run name": self.name,
                "Elapsed (seconds)": time_taken,
                "Training Accuracy %": np.round(100.0 * self.accuracy.best_accuracy, 2),
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
            this_summary_dataframe.to_markdown(buf=file)

        this_summary_dataframe.to_csv(self.csv_filepath, index=False)
