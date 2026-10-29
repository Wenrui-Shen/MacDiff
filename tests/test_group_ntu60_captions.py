"""Ensure grouping follows raw IDs and preserves every supplied sentence/person."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import group_ntu60_captions_by_class as group


class GroupCaptionsTests(unittest.TestCase):
    def person(self, slot, prefix):
        return {'person_index': slot, 'global': prefix + ' global',
                'local': {part: prefix + ' ' + part for part in group.LOCAL_PARTS}}

    def test_cli_groups_shuffled_raw_ids_and_preserves_all_people_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, source, output = root / 'train.npz', root / 'samples.json', root / 'by_class.txt'
            # x_train deliberately cannot be loaded with allow_pickle=False.
            np.savez(data, x_train=np.array([object()], dtype=object), y_train=np.eye(60)[[29, 7, 29]])
            rows = [dict(sample_index=2, persons=[self.person(1, 'two person1'), self.person(0, 'two person0')]),
                    dict(sample_index=0, persons=[self.person(0, 'zero')]),
                    dict(sample_index=1, persons=[self.person(0, 'one')])]
            rows[0]['persons'][1]['global'] = '  original\nspacing  '
            source.write_text(json.dumps(rows), encoding='utf-8')
            originals = (data.read_bytes(), source.read_bytes())
            args = group.get_parser().parse_args(['--data-path', str(data), '--captions', str(source), '--output', str(output)])
            with contextlib.redirect_stdout(io.StringIO()):
                group.run(args)
            text = output.read_text(encoding='utf-8')
            self.assertEqual(sum(line.startswith('A') and 'class_id_0based=' in line for line in text.splitlines()), 60)
            self.assertLess(text.index('sample_id=train_1 |'), text.index('sample_id=train_0 |'))
            self.assertLess(text.index('sample_id=train_0 |'), text.index('sample_id=train_2 |'))
            self.assertIn('A008 | class_id_0based=7 | sit down | samples=1', text)
            self.assertIn('A030 | class_id_0based=29 | type on a keyboard | samples=2', text)
            self.assertIn('person_index=1', text)
            for row in rows:
                for person in row['persons']:
                    for value in [person['global']] + list(person['local'].values()):
                        self.assertIn(json.dumps(value, ensure_ascii=False), text)
            self.assertEqual(originals, (data.read_bytes(), source.read_bytes()))
            with self.assertRaises(FileExistsError):
                group.run(args)

    def test_duplicate_missing_and_out_of_range_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'captions.json'
            first = dict(sample_index=0, persons=[self.person(0, 'text')])
            for rows, message in (([first, first], 'Duplicate'), ([first], 'Missing'),
                                  ([dict(first, sample_index=3)], 'outside')):
                source.write_text(json.dumps(rows), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, message):
                    group.group_captions([source], np.array([0, 1]))

    def test_multiple_jsonl_sources_and_original_slot_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / 'a.jsonl', Path(tmp) / 'b.jsonl']
            for i, path in enumerate(paths):
                path.write_text(json.dumps(dict(sample_index=i, persons=[self.person(1, str(i))])) + '\n', encoding='utf-8')
            result = group.group_captions(paths, np.array([51, 27]))
            self.assertEqual(result[51][0]['sample_index'], 0)
            self.assertEqual(result[27][0]['persons'][0]['person_index'], 1)
            self.assertEqual((group.ACTION_NAMES[27], group.ACTION_NAMES[51]), ('phone call', 'pushing'))

    def test_missing_body_text_and_non_ntu60_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, data = Path(tmp) / 'samples.json', Path(tmp) / 'data.npz'
            person = self.person(0, 'example')
            del person['local']['head']
            source.write_text(json.dumps([dict(sample_index=0, persons=[person])]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'six'):
                group.group_captions([source], np.array([0]))
            np.savez(data, y_train=np.eye(120)[[0, 1]])
            with self.assertRaisesRegex(ValueError, 'NTU60'):
                group.read_labels(data)


if __name__ == '__main__':
    unittest.main()
