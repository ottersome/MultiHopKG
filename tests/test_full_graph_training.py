from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_lp_training_data import build_full_graph


class FullGraphTrainingDataTests(unittest.TestCase):
    def test_build_full_graph_unions_and_deduplicates_all_kg_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / 'raw.kb').write_text('a b r1\nb c r2\n')
            (data_dir / 'train.triples').write_text('a b r1\nc d r3\n')
            (data_dir / 'dev.triples').write_text('d e r4\n')
            (data_dir / 'test.triples').write_text('e f r5\n')

            output_path = data_dir / 'lp_train.triples'
            self.assertEqual(build_full_graph(data_dir, output_path), 5)
            self.assertEqual(
                output_path.read_text().splitlines(),
                ['a\tb\tr1', 'b\tc\tr2', 'c\td\tr3', 'd\te\tr4', 'e\tf\tr5'],
            )


if __name__ == '__main__':
    unittest.main()
