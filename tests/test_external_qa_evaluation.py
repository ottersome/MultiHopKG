from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_utils import load_external_qa_test_data


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(token) for token in text.split()]


class ExternalQAEvaluationTests(unittest.TestCase):

    def _write_indices(self, directory: Path) -> tuple[Path, Path]:
        entities = directory / 'entity2id.txt'
        relations = directory / 'relation2id.txt'
        entities.write_text('source\t0\nanswer\t0\n')
        relations.write_text('relation\t0\n')
        return entities, relations

    def test_external_suite_is_test_only_and_uses_requested_question_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            entities, relations = self._write_indices(directory)
            source = directory / 'external.csv'
            pd.DataFrame([{
                'SplitLabel': 'test',
                'Setting': 'sa',
                'New-Paraphrase': 'fresh wording',
                'Source-Entity': 'source',
                'Answer-Entity': 'answer',
                'Paths': "[('source', 'relation', 'answer')]",
                'Path-Key': 'relation',
                'Hops': 1,
            }]).to_csv(source, index=False)
            cache = directory / 'external-cache.json'

            with patch('src.data_utils.AutoTokenizer.from_pretrained', return_value=_Tokenizer()):
                examples = load_external_qa_test_data(
                    str(source), str(cache), 'test-tokenizer', str(entities), str(relations),
                    question_column='New-Paraphrase', expected_setting='sa',
                )

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].question, [5, 7])
            self.assertEqual(examples[0].Paths, [[0, 0, 1]])
            self.assertEqual(examples[0].Path_Key, [0])
            self.assertTrue(cache.is_file())
            self.assertTrue(cache.with_suffix('.parquet').is_file())

            # A valid external cache must avoid retokenizing on subsequent runs.
            with patch('src.data_utils.AutoTokenizer.from_pretrained', side_effect=AssertionError):
                cached_examples = load_external_qa_test_data(
                    str(source), str(cache), 'test-tokenizer', str(entities), str(relations),
                    question_column='New-Paraphrase', expected_setting='sa',
                )
            self.assertEqual(cached_examples, examples)

    def test_external_suite_rejects_non_test_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            entities, relations = self._write_indices(directory)
            source = directory / 'external.csv'
            pd.DataFrame([{
                'SplitLabel': 'train',
                'Setting': 'sa',
                'New-Paraphrase': 'fresh wording',
                'Source-Entity': 'source',
                'Answer-Entity': 'answer',
                'Paths': "[('source', 'relation', 'answer')]",
                'Hops': 1,
            }]).to_csv(source, index=False)

            with self.assertRaisesRegex(ValueError, 'only SplitLabel=test'):
                load_external_qa_test_data(
                    str(source), str(directory / 'cache.json'), 'test-tokenizer',
                    str(entities), str(relations), expected_setting='sa',
                )

    def test_question_paraphrased_expands_variants_without_changing_row_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            entities, relations = self._write_indices(directory)
            source = directory / 'external.csv'
            pd.DataFrame([{
                'SplitLabel': 'test',
                'Setting': 'sa',
                'Question': 'original wording',
                'Question-Paraphrased': "['first variant', 'second variant']",
                'Source-Entity': 'source',
                'Answer-Entity': 'answer',
                'Paths': "[('source', 'relation', 'answer')]",
                'Hops': 1,
            }]).to_csv(source, index=False)

            with patch('src.data_utils.AutoTokenizer.from_pretrained', return_value=_Tokenizer()):
                examples = load_external_qa_test_data(
                    str(source), str(directory / 'cache.json'), 'test-tokenizer',
                    str(entities), str(relations),
                    question_column='Question-Paraphrased', expected_setting='sa',
                )

            self.assertEqual([example.question for example in examples], [[5, 7], [6, 7]])
            self.assertEqual([example.Eval_Weight for example in examples], [0.5, 0.5])


if __name__ == '__main__':
    unittest.main()
