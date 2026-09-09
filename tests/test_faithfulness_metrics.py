from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.eval import (
    FaithfulnessEvaluator,
    relation_edit_distance_raw,
    relation_overlap_f1,
)
from src.itl_typing import QAExample


class FaithfulnessMetricTests(unittest.TestCase):

    def test_relation_edit_distance_is_raw(self) -> None:
        self.assertEqual(
            relation_edit_distance_raw([1, 2, 3], [1, 4, 3], set(), {}),
            1,
        )

    def test_relation_overlap_uses_sets(self) -> None:
        _, _, f1 = relation_overlap_f1([1, 1, 2], [1, 2], set(), {})
        self.assertAlmostEqual(f1, 1.0, places=6)

    def test_semantic_paths_use_raw_kb_not_navigation_adjacency(self) -> None:
        kg = SimpleNamespace(
            entity2id={'source': 2, 'middle': 3, 'answer': 4},
            relation2id={'r1': 5, 'r2': 6},
            id2relation={5: 'r1', 6: 'r2'},
            adj_list={},
        )
        example = QAExample(2, [4], [], 2, Path_Key=[5, 6])
        with tempfile.TemporaryDirectory() as temporary_directory:
            Path(temporary_directory, 'raw.kb').write_text(
                'source middle r1\nmiddle answer r2\n'
            )
            evaluator = FaithfulnessEvaluator(
                kg, semantic_multi_path=True, data_dir=temporary_directory)
            paths = evaluator._get_semantically_valid_paths(example, [5, 6], [4])

        self.assertEqual(paths, [[(2, 5, 3), (3, 6, 4)]])

    def test_evaluator_reports_raw_relation_distance(self) -> None:
        kg = SimpleNamespace(
            entity2id={}, relation2id={}, id2relation={}, adj_list={},
        )
        evaluator = FaithfulnessEvaluator(kg)
        example = QAExample(2, 5, [], 3, Paths=[(2, 10, 3), (3, 11, 4), (4, 12, 5)])
        evaluator.update(example, [(2, 10, 3), (3, 20, 4), (4, 12, 5)], [5])
        self.assertEqual(evaluator.compute()['faithfulness/relation_edit_distance'], 1.0)


if __name__ == '__main__':
    unittest.main()
