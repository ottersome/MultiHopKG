import unittest

import pandas as pd

from src.data_utils import validate_question_lengths


class QuestionLengthValidationTest(unittest.TestCase):
    def test_includes_encoder_special_tokens(self):
        split, length = validate_question_lengths(
            [
                ('train', pd.DataFrame({'Question': [[1, 2, 3]]})),
                ('dev', pd.DataFrame({'Question': [[4]]})),
            ],
            max_question_len=5,
        )

        self.assertEqual(('train', 5), (split, length))

    def test_rejects_length_that_would_require_truncation(self):
        with self.assertRaisesRegex(AssertionError, 'would require truncation'):
            validate_question_lengths(
                [('test', pd.DataFrame({'Question': [[1, 2, 3, 4]]}))],
                max_question_len=5,
            )


if __name__ == '__main__':
    unittest.main()
