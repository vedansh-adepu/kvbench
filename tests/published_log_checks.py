"""Independent readback of the published tables and malformed-input cases."""
import copy
import csv
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
from reproduce_published_logs import check, memory_bounds, reproduce, sign_test


class PublishedLogs(unittest.TestCase):
    def setUp(self):
        self.result = reproduce(ROOT / 'docs/published-log-inputs.csv')
        self.document = (ROOT / 'docs/published-log-validation.md').read_text(encoding='utf-8')

    def test_published_output(self):
        check(self.result, self.document)
        self.assertEqual(self.result['counts'], [14, 13, 6])

    def test_first_row_drift_rejected(self):
        changed = copy.deepcopy(self.result)
        changed['intervals'][0][3] = '[0, 0]'
        with self.assertRaisesRegex(ValueError, 'interval table'):
            check(changed, self.document)

    def test_csv_public_provenance_and_settings(self):
        section = self.document.split('## Admitted inputs and public provenance\n', 1)[1].split('\n## ', 1)[0]
        with (ROOT / 'docs/published-log-inputs.csv').open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                expected = '| ' + ' | '.join([
                    f"[{row['record']}]({row['record_url']})", row['version'], row['model'],
                    row['runtime_dtype'], row['kv_dtype'], row['backend'], row['block'],
                    row['max_length'], row['tp']]) + ' |'
                self.assertIn(expected, section)
                self.assertIn(row['config_sha256'], self.document)

    def test_sign_drift_rejected(self):
        changed = copy.deepcopy(self.result)
        changed['signs'][0][0] = '12'
        with self.assertRaisesRegex(ValueError, 'sign or issue'):
            check(changed, self.document)

    def test_tie_boundaries(self):
        for rule in ['printf_2f', 'round_2_str']:
            low, high = memory_bounds('11.87', rule)
            display = lambda n: ('%.2f' % (n / 2**30)) if rule == 'printf_2f' else str(round(n / 2**30, 2))
            self.assertEqual(display(low), '11.87')
            self.assertEqual(display(high), '11.87')
            self.assertNotEqual(display(low - 1), '11.87')
            self.assertNotEqual(display(high + 1), '11.87')

    def test_invalid_input(self):
        for gib, rule in [('NaN', 'printf_2f'), ('0', 'printf_2f'), ('1', 'unknown')]:
            with self.assertRaises(ValueError):
                memory_bounds(gib, rule)
        with patch.object(Path, 'open', return_value=io.StringIO('record\n')):
            with self.assertRaises(ValueError):
                reproduce(Path('unused.csv'))

    def test_zero_signs_excluded(self):
        self.assertEqual(sign_test([0, -1, 1]), ['1', '1', '1', '1 = 1.0'])


if __name__ == '__main__':
    unittest.main()
