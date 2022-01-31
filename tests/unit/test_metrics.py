import os
import unittest
from adaswarm.utils import Metrics
from unittest.mock import patch
import pandas as pd
from freezegun import freeze_time


MD_OUTPUT_FILENAME = "./tests/unit/output/summary.md"
CSV_OUTPUT_FILENAME = "./tests/unit/output/summary.csv"


class TestMetrics(unittest.TestCase):
    @staticmethod
    def read_dataframe_from_markdown():
        dataframe = (
            pd.read_table(
                MD_OUTPUT_FILENAME,
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

    @staticmethod
    def read_dataframe_from_csv():
        dataframe = pd.read_csv(CSV_OUTPUT_FILENAME, index_col=False)

        dataframe.columns = [col.strip() for col in dataframe.columns]
        return dataframe

    def tearDown(self):
        try:
            if os.path.exists(MD_OUTPUT_FILENAME):
                os.remove(MD_OUTPUT_FILENAME)
            if os.path.exists(CSV_OUTPUT_FILENAME):
                os.remove(CSV_OUTPUT_FILENAME)
        except OSError as oserr:
            print(oserr)

    def test_train_accuracy(self):
        with Metrics(
            md_filepath=MD_OUTPUT_FILENAME, csv_filepath=CSV_OUTPUT_FILENAME
        ) as metrics:
            metrics.update_training_accuracy(1.5)
            metrics.update_training_accuracy(1.0)
            self.assertEqual(metrics.best_training_accuracy, 1.5)

    def test_train_accuracy_md_output(self):
        with Metrics(
            md_filepath=MD_OUTPUT_FILENAME, csv_filepath=CSV_OUTPUT_FILENAME
        ) as metrics:
            metrics.update_training_accuracy(0.01523)
        self.assertEqual(
            float(self.read_dataframe_from_markdown()["Training Accuracy %"].values[0]),
            1.52,
        )

    def test_train_accuracy_csv_output(self):
        with Metrics(
            md_filepath=MD_OUTPUT_FILENAME, csv_filepath=CSV_OUTPUT_FILENAME
        ) as metrics:
            metrics.update_training_accuracy(0.01523)
        self.assertEqual(
            float(self.read_dataframe_from_csv()["Training Accuracy %"].values[0]), 1.52
        )

    def test_time_taken(self):
        with patch("time.time", return_value=1):
            with Metrics(
                md_filepath=MD_OUTPUT_FILENAME, csv_filepath=CSV_OUTPUT_FILENAME
            ):
                self.assertEqual(True, True)

            self.assertEqual(
                int(self.read_dataframe_from_markdown()["Elapsed (seconds)"].values[0]),
                0,
            )

    def test_train_accuracy_csv_output(self):
        with Metrics(
            md_filepath=MD_OUTPUT_FILENAME, csv_filepath=CSV_OUTPUT_FILENAME
        ) as metrics:
            metrics.update_training_accuracy(0.01523)

        with Metrics(
            md_filepath=MD_OUTPUT_FILENAME, csv_filepath=CSV_OUTPUT_FILENAME
        ) as metrics:
            metrics.update_training_accuracy(0.01623)
        self.assertEqual(
            float(self.read_dataframe_from_csv()["Training Accuracy %"].values[0]), 1.52
        )
        self.assertEqual(
            float(self.read_dataframe_from_csv()["Training Accuracy %"].values[1]), 1.62
        )

    @freeze_time("2021-12-25 03:01:33")
    def test_start_time_in_output(self):

        with Metrics(
            md_filepath=MD_OUTPUT_FILENAME, csv_filepath=CSV_OUTPUT_FILENAME
        ) as metrics:
            metrics.update_training_accuracy(0.01523)

        self.assertEqual(
            self.read_dataframe_from_csv()["Start time"].values[0],
            "25-12-21 03:01:33",
        )

    def test_epoch(self):

        with Metrics(
            md_filepath=MD_OUTPUT_FILENAME, csv_filepath=CSV_OUTPUT_FILENAME
        ) as metrics:
            metrics.current_epoch(2)

        self.assertEqual(
            self.read_dataframe_from_csv()["Number of epochs"].values[0],
            2,
        )

    def test_best_loss(self):

        with Metrics(
            md_filepath=MD_OUTPUT_FILENAME, csv_filepath=CSV_OUTPUT_FILENAME
        ) as metrics:
            metrics.update_training_loss(0.2)
            metrics.update_training_loss(0.1)
            metrics.update_training_loss(0.3)

        self.assertEqual(
            self.read_dataframe_from_csv()["Training Loss"].values[0],
            0.1,
        )

    def test_empty_best_loss(self):

        with Metrics(md_filepath=MD_OUTPUT_FILENAME, csv_filepath=CSV_OUTPUT_FILENAME):
            pass

        self.assertEqual(
            self.read_dataframe_from_csv()["Training Loss"].values[0],
            "Not set",
        )
