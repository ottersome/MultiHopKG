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
from src.data_utils import load_full_graph_adjacency_matrix
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

    def test_semantic_paths_use_full_explicit_graph_not_raw_kb(self) -> None:
        kg = SimpleNamespace(
            entity2id={'source': 2, 'middle': 3, 'answer': 4},
            relation2id={'r1': 5, 'r2': 6},
            id2relation={5: 'r1', 6: 'r2'},
            adj_list={},
        )
        example = QAExample(2, [4], [], 2, Path_Key=[5, 6])
        with tempfile.TemporaryDirectory() as temporary_directory:
            Path(temporary_directory, 'raw.kb').write_text('')
            Path(temporary_directory, 'train.triples').write_text('source middle r1\n')
            Path(temporary_directory, 'dev.triples').write_text('middle answer r2\n')
            Path(temporary_directory, 'test.triples').write_text('')
            Path(temporary_directory, 'entity2id.txt').write_text(
                'dummy0\t0\ndummy1\t0\nsource\t0\nmiddle\t0\nanswer\t0\n')
            Path(temporary_directory, 'relation2id.txt').write_text(
                'dummy0\t0\ndummy1\t0\ndummy2\t0\ndummy3\t0\ndummy4\t0\nr1\t0\nr2\t0\n')
            evaluator = FaithfulnessEvaluator(
                kg, semantic_multi_path=True, data_dir=temporary_directory)
            paths = evaluator._get_semantically_valid_paths(example, [5, 6], [4])
            evaluator.update(example, [(2, 5, 3), (3, 6, 4)], [4])

        self.assertEqual(paths, [[(2, 5, 3), (3, 6, 4)]])
        metrics = evaluator.compute()
        self.assertEqual(metrics['faithfulness/semantic_path_attempts'], 1.0)
        self.assertEqual(metrics['faithfulness/semantic_path_examples'], 1.0)
        self.assertEqual(metrics['faithfulness/semantic_path_coverage'], 1.0)

    def test_full_graph_adjacency_is_reused_between_evaluators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for split in ('train', 'dev', 'test'):
                (directory / '{}.triples'.format(split)).write_text('source answer r\n')
            (directory / 'entity2id.txt').write_text('source\t0\nanswer\t0\n')
            (directory / 'relation2id.txt').write_text('r\t0\n')
            first = load_full_graph_adjacency_matrix(directory)
            for split in ('train', 'dev', 'test'):
                (directory / '{}.triples'.format(split)).unlink()
            second = load_full_graph_adjacency_matrix(directory)

        self.assertIs(first, second)

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
