"""Group every NTU60 XSub training caption into one text file by action class.

Uses y_train and original sample_index, without loading skeleton arrays or models.
Preserves all supplied people and all seven sentences. No similarity statistics.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from util.structured_text_cache import LOCAL_PARTS


# Official NTU RGB+D A1-A60: https://rose1.ntu.edu.sg/dataset/actionRecognition/
ACTION_NAMES = (
    'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop', 'pick up',
    'throw', 'sit down', 'stand up', 'clapping', 'reading', 'writing',
    'tear up paper', 'put on jacket', 'take off jacket', 'put on a shoe',
    'take off a shoe', 'put on glasses', 'take off glasses', 'put on a hat/cap',
    'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something',
    'reach into pocket', 'hopping', 'jump up', 'phone call', 'play with phone/tablet',
    'type on a keyboard', 'point to something', 'taking a selfie',
    'check time (from watch)', 'rub two hands', 'nod head/bow', 'shake head',
    'wipe face', 'salute', 'put palms together', 'cross hands in front',
    'sneeze/cough', 'staggering', 'falling down', 'headache', 'chest pain',
    'back pain', 'neck pain', 'nausea/vomiting', 'fan self', 'punch/slap',
    'kicking', 'pushing', 'pat on back', 'point finger', 'hugging', 'giving object',
    'touch pocket', 'shaking hands', 'walking towards', 'walking apart',
)


def read_labels(path):
    # NPZ members are lazy: do not access or decompress x_train.
    with np.load(path, allow_pickle=False) as data:
        labels = data['y_train']
    if (labels.ndim != 2 or labels.shape[1] != 60 or not len(labels)
            or not np.isfinite(labels).all()
            or not np.all((labels > 0).sum(axis=1) == 1)):
        raise ValueError('Expected NTU60 y_train: [samples,60], one positive class per row')
    return labels.argmax(axis=1)


def iter_records(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        if Path(path).suffix.lower() == '.json':
            records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError('Expected a caption JSON array: ' + str(path))
            yield from records
        else:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def group_captions(paths, labels):
    grouped, seen = [[] for _ in range(60)], set()
    for path in paths:
        for record in iter_records(path):
            if not isinstance(record, dict):
                raise ValueError('Expected a caption record object')
            index = record.get('sample_index')
            if type(index) is not int or not 0 <= index < len(labels):
                raise ValueError('Caption sample_index is outside y_train: ' + repr(index))
            if index in seen:
                raise ValueError('Duplicate sample_index; use each caption source once: ' + str(index))
            if record.get('source_split', 'train') != 'train':
                raise ValueError('Only training captions are supported: ' + str(index))
            people = record.get('persons')
            if not isinstance(people, list) or not 1 <= len(people) <= 2:
                raise ValueError('Expected one or two original person slots: ' + str(index))
            slots = set()
            for person in people:
                if (not isinstance(person, dict)
                        or set(person) != {'person_index', 'global', 'local'}):
                    raise ValueError('Expected v3 global/local person captions: ' + str(index))
                slot = person['person_index']
                if type(slot) is not int or slot not in (0, 1) or slot in slots:
                    raise ValueError('Invalid/duplicate original person index: ' + str(index))
                slots.add(slot)
                local = person['local']
                if not isinstance(local, dict) or set(local) != set(LOCAL_PARTS):
                    raise ValueError('Expected all six body-region texts: ' + str(index))
                texts = [person['global']] + [local[part] for part in LOCAL_PARTS]
                if any(not isinstance(text, str) or not text.strip() for text in texts):
                    raise ValueError('Empty/malformed caption text: ' + str(index))
            grouped[int(labels[index])].append(record)
            seen.add(index)
    missing = sorted(set(range(len(labels))) - seen)
    if missing:
        raise ValueError('Missing {} training captions (first 20 indices): {}'.format(len(missing), missing[:20]))
    for rows in grouped:
        rows.sort(key=lambda row: row['sample_index'])
    return grouped


def write_grouped(path, grouped, data_path, caption_paths):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Choose a new --output; existing files are not overwritten: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('NTU60 XSub training captions grouped by action class\n')
        stream.write('data_path: ' + str(Path(data_path).resolve()) + '\n')
        for source in caption_paths:
            stream.write('caption_source: ' + str(Path(source).resolve()) + '\n')
        stream.write('Class IDs use y_train: 0..59; action codes A001..A060 use ID+1.\n')
        stream.write('All original person slots and global + six local sentences are retained.\n')
        stream.write('Strings are JSON-quoted to preserve whitespace/newlines without rewriting text.\n')
        for class_id, rows in enumerate(grouped):
            stream.write('\n' + '=' * 88 + '\n')
            stream.write('A{:03d} | class_id_0based={} | {} | samples={}\n'.format(
                class_id + 1, class_id, ACTION_NAMES[class_id], len(rows)))
            stream.write('=' * 88 + '\n')
            for record in rows:
                index = record['sample_index']
                stream.write('\nsample_id=train_{} | sample_index={}\n'.format(index, index))
                if 'sample_id' in record:
                    stream.write('source_sample_id: ' + json.dumps(record['sample_id'], ensure_ascii=False) + '\n')
                for person in sorted(record['persons'], key=lambda row: row['person_index']):
                    stream.write('  person_index={}\n'.format(person['person_index']))
                    stream.write('    global: ' + json.dumps(person['global'], ensure_ascii=False) + '\n')
                    for part in LOCAL_PARTS:
                        stream.write('    {}: {}\n'.format(part, json.dumps(person['local'][part], ensure_ascii=False)))


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='../data/MAMP/ntu/NTU60_XSub.npz')
    parser.add_argument('--captions', nargs='+',
                        default=['vlm_pilot/ntu60_xsub_clip_sentence_cache_v3/samples.json'],
                        help='Complete v3 samples.json/caption JSON array or non-overlapping JSONL shards')
    parser.add_argument('--output', default='output_dir/ntu60_xsub_captions_by_class.txt')
    return parser


def run(args):
    if Path(args.output).exists():
        raise FileExistsError('Choose a new --output; existing files are not overwritten: ' + args.output)
    labels = read_labels(args.data_path)
    grouped = group_captions(args.captions, labels)
    write_grouped(args.output, grouped, args.data_path, args.captions)
    print('Grouped {} samples into 60 class sections: {}'.format(len(labels), Path(args.output).resolve()))


if __name__ == '__main__':
    run(get_parser().parse_args())
