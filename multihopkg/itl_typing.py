from typing import List, Tuple, Dict, Any, DefaultDict
from collections import namedtuple
from dataclasses import dataclass
import pandas as pd

Triple = Tuple[int, int, int]
Triples = List[Triple]
# Named Tuple for DF SPlit
SplitTuple = namedtuple("SplitTuple", ["train", "dev", "test"])

@dataclass
class DFSplit:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame

