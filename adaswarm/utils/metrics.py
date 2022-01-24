import time

class Timer(object):
    def __init__(self, name="Time taken"):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, *args):
        print(f"""
------------------------------------------
            { self.name }
------------------------------------------
Elapsed: {(time.time() - self.tstart)}
------------------------------------------
        """)

class Stat(object):
    class Accuracy(object):
        def __init__(self):
            self.best_accuracy = 0.0

        def update(self, value):
            if value > self.best_accuracy:
                self.best_accuracy = value 

    def __init__(self, name="Time taken"):
        self.name = name
        self.accuracy = Stat.Accuracy()

    def __enter__(self):
        return self.accuracy

    def __exit__(self, *args):
        pass

