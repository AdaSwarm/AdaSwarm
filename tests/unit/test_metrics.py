import unittest
from adaswarm.utils import Stat, Timer
from unittest.mock import patch
import pandas as pd


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        with Stat() as stat:
            stat.update(1.5)
            stat.update(1.0)
            self.assertEqual(stat.best_accuracy, 1.5)

    def test_time_taken(self):
        with patch("time.time", return_value=1):
            with Timer():
                self.assertEqual(True, True)

            dataframe = (
                pd.read_table(
                    "tests/unit/output/summary.md",
                    sep="|",
                    header=0,
                    index_col=1,
                    skipinitialspace=True,
                )
                .dropna(axis=1, how="all")
                .iloc[1:]
            )

            dataframe.columns = [col.strip() for col in dataframe.columns]

            self.assertEqual(int(dataframe["Elapsed (seconds)"].values[0]), 0)
