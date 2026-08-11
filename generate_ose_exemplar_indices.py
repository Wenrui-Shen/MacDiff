"""Generate one reproducible OSE exemplar dataset index per training class."""

import argparse
import json
from pathlib import Path

import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Select one immutable training-set index per class from a MAMP '
            'NPZ archive and write the OSE exemplar JSON mapping.'))
    parser.add_argument(
        '--data_path',
        default='../data/MAMP/ntu/NTU60_XSub.npz',
        help='MAMP NPZ archive containing x_train and one-hot y_train')
    parser.add_argument(
        '--output_path',
        default='config/ntu60_xsub_joint/exemplar_indices.json',
        help='Destination JSON path')
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Random seed used to choose one sample within each class')
    return parser


def load_training_labels(data_path):
    archive = np.load(data_path, mmap_mode='r')
    required = {'x_train', 'y_train'}
    missing = sorted(required - set(archive.files))
    if missing:
        raise KeyError(
            '{} is missing required arrays: {}'.format(data_path, missing))

    targets = archive['y_train']
    if targets.ndim != 2:
        raise ValueError(
            'y_train must be a two-dimensional one-hot array, got shape {}'.format(
                targets.shape))
    positive = targets > 0
    positives_per_sample = positive.sum(axis=1)
    if not np.all(positives_per_sample == 1):
        bad = int(np.count_nonzero(positives_per_sample != 1))
        raise ValueError(
            'y_train must contain exactly one positive class per sample; '
            '{} rows violate this requirement'.format(bad))
    if archive['x_train'].shape[0] != targets.shape[0]:
        raise ValueError(
            'x_train and y_train must contain the same number of samples')

    # This is exactly the label conversion used by feeder.feeder_ntu.Feeder.
    labels = np.where(positive)[1].astype(np.int64, copy=False)
    return labels


def select_exemplars(labels, seed):
    classes = np.unique(labels)
    expected = np.arange(int(classes[-1]) + 1, dtype=classes.dtype)
    if not np.array_equal(classes, expected):
        raise ValueError(
            'Training class IDs must be contiguous from 0; found {}'.format(
                classes.tolist()))

    rng = np.random.RandomState(seed)
    mapping = {}
    for class_id in classes.tolist():
        candidates = np.flatnonzero(labels == class_id)
        if candidates.size == 0:
            raise RuntimeError('No training sample for class {}'.format(class_id))
        mapping[str(int(class_id))] = int(rng.choice(candidates))

    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError('Selected exemplar indices are unexpectedly duplicated')
    if any(int(labels[index]) != int(class_id)
           for class_id, index in mapping.items()):
        raise RuntimeError('Generated exemplar mapping failed label validation')
    return mapping


def main():
    args = get_args_parser().parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    if not data_path.is_file():
        raise FileNotFoundError('MAMP archive does not exist: {}'.format(data_path))

    labels = load_training_labels(data_path)
    mapping = select_exemplars(labels, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(mapping, file, indent=2, sort_keys=True)
        file.write('\n')

    print('Data:', data_path)
    print('Seed:', args.seed)
    print('Classes:', len(mapping))
    print('Output:', output_path)
    print('Exemplar indices:', mapping)


if __name__ == '__main__':
    main()
