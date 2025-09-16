"""
Type definitions and data structures for the MINERVA project.

This module provides common type aliases and data structures used throughout
the project for knowledge graph operations and data handling. It defines
standardized representations for triples, dataset splits, and other core
data structures to ensure type safety and consistency across the codebase.

Type Aliases:
    Triple: Represents a knowledge graph triple (head, relation, tail) as entity/relation IDs
    Triples: Collection of multiple Triple instances
    
Data Structures:
    SplitTuple: Named tuple for dataset splits (train, dev, test)
    DFSplit: Dataclass for pandas DataFrame dataset splits with type safety
"""

import pandas as pd
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple

# Knowledge Graph Triple Types
Triple = Tuple[int, int, int]
"""
Type alias for a knowledge graph triple.

Represents a single knowledge graph fact as (head_entity_id, relation_id, tail_entity_id).
All components are integer IDs corresponding to entities and relations in the vocabulary.

Example:
    A triple (5, 12, 23) might represent "Person_5 works_for Company_23"
    where 5 is the head entity ID, 12 is the relation ID, and 23 is the tail entity ID.
"""

Triples = List[Triple]
"""
Type alias for a collection of knowledge graph triples.

Represents multiple Triple instances, typically used for datasets, paths,
or collections of facts in knowledge graph operations.
"""

# Dataset Split Types
SplitTuple = namedtuple("SplitTuple", ["train", "dev", "test"])
"""
Named tuple for organizing dataset splits.

Provides a simple structure for grouping train, development, and test datasets
with named access. This is a lightweight alternative to the DFSplit dataclass
for cases where pandas DataFrames are not required.

Attributes:
    train: Training dataset
    dev: Development/validation dataset  
    test: Test dataset
"""

@dataclass
class DFSplit:
    """
    Dataclass for organizing pandas DataFrame dataset splits with type safety.
    
    Provides a structured way to handle train/dev/test splits of pandas DataFrames
    with explicit type annotations and automatic validation. This is preferred
    over SplitTuple when working with pandas DataFrames as it provides better
    type checking and IDE support.
    
    Attributes:
        train (pd.DataFrame): Training dataset DataFrame
        dev (pd.DataFrame): Development/validation dataset DataFrame
        test (pd.DataFrame): Test dataset DataFrame
        
    Example:
        >>> import pandas as pd
        >>> train_df = pd.DataFrame({'question': ['What is X?'], 'answer': ['Y']})
        >>> dev_df = pd.DataFrame({'question': ['What is Z?'], 'answer': ['W']})
        >>> test_df = pd.DataFrame({'question': ['What is A?'], 'answer': ['B']})
        >>> split = DFSplit(train=train_df, dev=dev_df, test=test_df)
        >>> print(len(split.train))  # Access training data
        1
    """
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
