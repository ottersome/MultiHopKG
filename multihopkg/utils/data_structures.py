from dataclasses import dataclass
from typing import Tuple, Dict
import pandas as pd

class DataPartitions:

    ASSUMED_COLUMNS = ["enc_questions", "enc_answer", "triples_ints"]
    def __init__(self, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):

        # Ensure integrity of dataset
        for ac in self.ASSUMED_COLUMNS:
            for ds in [train,validation, test]:
                if ac not in ds.columns:
                    error_str = f"Expected column '{ac}' to be found in the dataset."\
                        f"But instead we get columns {ds.columns}"
                    raise ValueError(error_str)

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

# Mostly for convenience of moving data around and keeping the code "typed"
Triplet_Str = Tuple[str, str, str]
Triplet_Int = Tuple[int, int, int]

@dataclass
class UniversalIdxDictionary:
    """
    Meant to be a convenience to hold bidirectional dictionaries for both entities and relations
    """
    id2entity: Dict[int, str]
    entity2id: Dict[str, int]
    id2relation: Dict[int, str]
    relation2id: Dict[str, int]
