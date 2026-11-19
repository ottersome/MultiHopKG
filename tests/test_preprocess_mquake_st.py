from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.preprocess_mquake_st import build_single_answer_qa_file
from scripts.preprocess_mquake_st_common import build_mquake_st_kg_files
from scripts.preprocess_mquake_st_ma import build_multi_answer_qa_file


class MQuakePreprocessingTests(unittest.TestCase):

    def test_shared_kg_and_qa_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            kg_dir = root / 'kg'
            kg_dir.mkdir()
            (kg_dir / 'train.txt').write_text('tail_train rel head_train\n')
            (kg_dir / 'valid.txt').write_text('tail_valid rel head_valid\n')
            (kg_dir / 'test.txt').write_text('tail_test rel head_test\n')
            output_dir = root / 'data'
            output_path = build_mquake_st_kg_files(kg_dir / 'triplets.txt', output_dir / 'raw.kb')

            self.assertEqual(
                (output_dir / 'raw.kb').read_text().splitlines(),
                ['tail_train\thead_train\trel', 'tail_valid\thead_valid\trel', 'tail_test\thead_test\trel'],
            )
            self.assertEqual(
                (output_dir / 'train.triples').read_text().splitlines(),
                ['tail_train\thead_train\trel'],
            )

            single_source = root / 'single.csv'
            pd.DataFrame([{
                'Question-Number': 17,
                'Question': 'single?',
                'Answer': 'answer',
                'Hops': 1,
                'Source-Entity': 'source',
                'Answer-Entity': 'answer_entity',
                'Paths': "[('source', 'rel', 'answer_entity')]",
                'Path-Key': 'rel',
                'SplitLabel': 'test',
            }]).to_csv(single_source, index=False)
            build_single_answer_qa_file(single_source, output_path, 'single.csv')
            single_output = pd.read_csv(output_dir / 'single.csv')
            self.assertEqual(single_output.loc[0, 'Question-Number'], 17)
            self.assertNotIn('Unnamed: 0', single_output.columns)
            self.assertIn('Paths', single_output.columns)

            multi_source = root / 'multi.csv'
            pd.DataFrame([{
                'Question-Number': 18,
                'Question': 'multi?',
                'Answer': "['answer_a', 'answer_b']",
                'Hops': 1,
                'Source-Entity': 'source',
                'Answer-Entity': "['answer_a', 'answer_b']",
                'Path-Key': 'rel',
                'SplitLabel': 'test',
            }]).to_csv(multi_source, index=False)
            build_multi_answer_qa_file(multi_source, output_path, 'multi.csv')
            multi_output = pd.read_csv(output_dir / 'multi.csv')
            self.assertEqual(multi_output.loc[0, 'Question-Number'], 18)
            self.assertNotIn('Paths', multi_output.columns)
            self.assertEqual(multi_output.loc[0, 'Answer-Entity'], "['answer_a', 'answer_b']")


if __name__ == '__main__':
    unittest.main()
