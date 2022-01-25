import os
import time
import pandas as pd


class Metrics(object):
    class Stats(object):
        def __init__(self):
            self.best_accuracy = 0.0

        def update_train_accuracy(self, value):
            if value > self.best_accuracy:
                self.best_accuracy = value

    def __init__(self, name="Default run"):
        self.name = name
        self.accuracy = Metrics.Stats()
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        return self.accuracy

    def __exit__(self, *args):

        time_taken = (time.time() - self.tstart)
        this_summary_dataframe = pd.DataFrame({
            "Run name":self.name, 
            "Elapsed (seconds)":time_taken, 
            "Training Accuracy":self.accuracy.best_accuracy}, 
            index=[0])

        os.makedirs(os.path.dirname("tests/unit/output/"), exist_ok=True)

        with open("tests/unit/output/summary.md", "w") as file:
            this_summary_dataframe.to_markdown(buf=file)

