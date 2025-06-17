"""
Mostly for convenience of moving data around and keeping the code "typed"
"""
import pandas as pd

class DataPartitions:
    def __init__(self, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
        self._train = train
        self._validation = validation
        self._test = test

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, data):
        self._train = data

    @property
    def validation(self):
        return self._validation

    @validation.setter
    def validation(self, data):
        self._validation = data

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, data):
        self._test = data
