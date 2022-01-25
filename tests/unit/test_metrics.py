import os
import unittest
from adaswarm.utils import Metrics 
from unittest.mock import patch
import pandas as pd


OUTPUT_FILENAME = "./tests/unit/output/summary.md"

class TestMetrics(unittest.TestCase):
    @staticmethod
    def read_dataframe():
            dataframe = (
                pd.read_table(
                    OUTPUT_FILENAME,
                    sep="|",
                    header=0,
                    index_col=1,
                    skipinitialspace=True,
                )
                .dropna(axis=1, how="all")
                .iloc[1:]
            )

            dataframe.columns = [col.strip() for col in dataframe.columns]
            return dataframe

    def tearDown(self):
        try:
            os.remove(OUTPUT_FILENAME)
        except OSError as oserr:
            print(oserr)



    def test_train_accuracy(self):
        with Metrics(filepath=OUTPUT_FILENAME) as metrics:
            metrics.update_train_accuracy(1.5)
            metrics.update_train_accuracy(1.0)
            self.assertEqual(metrics.best_accuracy, 1.5)


    def test_train_accuracy_output(self):
        with Metrics(filepath=OUTPUT_FILENAME) as metrics:
            metrics.update_train_accuracy(0.01523)
        self.assertEqual(float(self.read_dataframe()["Training Accuracy"].values[0]), 1.52)

    def test_time_taken(self):
        with patch("time.time", return_value=1):
            with Metrics(filepath=OUTPUT_FILENAME):
                self.assertEqual(True, True)

            self.assertEqual(int(self.read_dataframe()["Elapsed (seconds)"].values[0]), 0)
