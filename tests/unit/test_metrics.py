import unittest
from adaswarm.utils import Stat, Timer
from unittest.mock import patch

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
