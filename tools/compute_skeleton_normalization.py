#!/usr/bin/env python3
"""Audit train-only coordinate statistics against PT/LP config constants.

NumPy suffices for raw statistics. YAML configs require PyYAML; the optional
model-input view calls the repository's CPU Torch preprocessing functions.
NPZ x_train is streamed, rather than loaded whole by np.load(mmap_mode=...).
"""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'macdiff.train_coordinate_normalization_audit.v1'
DEFAULT_CONFIG = 'config/ntu60_xsub_joint/pretrain_madiff_flow_matching.yaml'


class TrainArray:
    """Read x_train only; bound resident array memory by chunk_size samples."""
    def __init__(self, path):
        self.path = Path(path).resolve()

    def __enter__(self):
        self.stack = contextlib.ExitStack()
        try:
            if self.path.suffix.lower() == '.npz':
                archive = self.stack.enter_context(zipfile.ZipFile(self.path))
                member = archive.getinfo('x_train.npy')
                self.stream = self.stack.enter_context(archive.open(member))
                version = np.lib.format.read_magic(self.stream)
                if version == (1, 0):
                    header = np.lib.format.read_array_header_1_0(self.stream)
                elif version == (2, 0):
                    header = np.lib.format.read_array_header_2_0(self.stream)
                else:
                    raise ValueError('Unsupported NPY header version: %r' % (version,))
                self.shape, fortran, self.dtype = header
                if fortran:
                    raise ValueError('Fortran-order NPZ: extract x_train.npy and pass that NPY instead')
                self.array = None
                identity = dict(member='x_train.npy', member_crc32=member.CRC,
                                member_bytes=member.file_size)
            elif self.path.suffix.lower() == '.npy':
                self.array = np.load(self.path, mmap_mode='r', allow_pickle=False)
                self.stack.callback(self.array._mmap.close)
                self.shape, self.dtype = self.array.shape, self.array.dtype
                identity = dict(member='user_supplied_train_npy')
            else:
                raise ValueError('Data must be an NPZ with x_train, or a training-only NPY')
            if len(self.shape) not in (3, 5) or self.dtype.kind not in 'fiu':
                raise ValueError('Expected a numeric [N,T,M*V*3] or five-dimensional training array')
            stat = self.path.stat()
            self.metadata = dict(path=str(self.path), shape=list(self.shape), dtype=str(self.dtype),
                                 file_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, **identity)
            return self
        except BaseException:
            self.stack.close()
            raise

    def batches(self, chunk_size, limit=0):
        count = min(limit, self.shape[0]) if limit else self.shape[0]
        sample_values = int(np.prod(self.shape[1:], dtype=np.int64))
        for start in range(0, count, chunk_size):
            size = min(chunk_size, count - start)
            if self.array is not None:
                batch = self.array[start:start + size]
            else:
                required = size * sample_values * self.dtype.itemsize
                payload = self.stream.read(required)
                if len(payload) != required:
                    raise ValueError('Truncated x_train at sample %d' % start)
                batch = np.frombuffer(payload, dtype=self.dtype).reshape((size,) + self.shape[1:])
            yield start, batch

    def __exit__(self, *exception):
        return self.stack.__exit__(*exception)


def sample_ctvm(sample, joints=25, layout='auto'):
    """Flattened storage uses the same T,M,V,C ordering as feeder_ntu.Feeder."""
    if layout == 'auto':
        if sample.ndim == 2:
            layout = 'flat'
        elif sample.ndim == 4:
            ctvm = sample.shape[0] == 3 and sample.shape[2] == joints
            tmvc = sample.shape[-1] == 3 and sample.shape[2] == joints
            if ctvm == tmvc:
                raise ValueError('Ambiguous five-dimensional layout; set --layout explicitly')
            layout = 'ctvm' if ctvm else 'tmvc'
    if layout == 'flat' and sample.ndim == 2:
        frames, features = sample.shape
        if features % (joints * 3):
            raise ValueError('Flattened feature count must be divisible by joints*3')
        result = sample.reshape(frames, features // (joints * 3), joints, 3).transpose(3, 0, 2, 1)
    elif layout == 'ctvm' and sample.ndim == 4:
        result = sample
    elif layout == 'tmvc' and sample.ndim == 4:
        result = sample.transpose(3, 0, 2, 1)
    else:
        raise ValueError('Unsupported skeleton shape/layout: %r / %s' % (sample.shape, layout))
    if result.shape[0] != 3 or result.shape[2] != joints or result.shape[-1] < 1:
        raise ValueError('Expected nonempty C=3,T,V=%d,M skeletons' % joints)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite coordinates in training data')
    return result


class Moments:
    """Merge float64 central moments; never concatenate the whole dataset."""
    def __init__(self):
        self.count = 0
        self.mean = np.zeros(3, dtype=np.float64)
        self.m2 = np.zeros(3, dtype=np.float64)

    def update(self, coordinates):
        values = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        count = len(values)
        if not count:
            return
        mean = values.mean(axis=0)
        m2 = ((values - mean) ** 2).sum(axis=0)
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta ** 2 * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def result(self):
        if not self.count:
            return dict(count=0, mean=None, var=None, std=None)
        variance = np.maximum(self.m2 / self.count, 0.)
        return dict(count=self.count, mean=self.mean.tolist(), var=variance.tolist(),
                    std=np.sqrt(variance).tolist())


def update_groups(groups, sample, view, person_modes):
    # Test occupancy BEFORE centering: a missing frame can become nonzero after
    # self_shift. Keep individual zero coordinates and joints in occupied frames.
    active = np.any(sample != 0, axis=(0, 2))  # [T,M]
    for shifted in (False, True):
        values = np.array(sample, dtype=np.float64, copy=True)
        if shifted:
            values -= values.mean(axis=(1, 2), keepdims=True)  # independently per person
        for people in person_modes:
            subset = values[..., :1] if people == 'person0' else values
            occupied = active[:, :1] if people == 'person0' else active
            coordinates = subset.transpose(1, 3, 2, 0)  # [T,M,V,C]
            for padding in ('model_positions', 'active_person_frames'):
                key = (view, people, shifted, padding)
                selected = coordinates if padding == 'model_positions' else coordinates[occupied]
                groups.setdefault(key, Moments()).update(selected)


def load_config(path):
    path = Path(path).resolve()
    content = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        config = json.loads(content)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError('Reading YAML requires the repository dependency PyYAML') from exc
        config = yaml.safe_load(content)
    if not isinstance(config, dict) or not isinstance(config.get('model_args'), dict):
        raise ValueError('Config must have a model_args mapping: %s' % path)
    return config, dict(path=str(path), sha256=hashlib.sha256(content.encode('utf-8')).hexdigest())


def config_reference(config, identity):
    options = config['model_args']
    mean = np.asarray(options.get('input_mean', [0., 0., 0.]), dtype=np.float64)
    variance = np.asarray(options.get('input_var', [1., 1., 1.]), dtype=np.float64)
    if mean.shape != (3,) or variance.shape != (3,) or not np.isfinite(mean).all() or not (
            np.isfinite(variance).all() and np.all(variance > 0)):
        raise ValueError('Config input_mean/input_var must be finite XYZ vectors, with positive variance')
    shifted = options.get('self_shift', False)
    default_person0 = 'transformer_downstream' not in config.get('model', '')
    person0 = options.get('one_person', default_person0)
    if not isinstance(shifted, bool) or not isinstance(person0, bool):
        raise ValueError('self_shift and one_person must be booleans')
    return dict(**identity, configured_mean=mean.tolist(), var=variance.tolist(),
                effective_mean=([0., 0., 0.] if shifted else mean.tolist()),
                self_shift=shifted, person_mode='person0' if person0 else 'all')


def compare_statistics(statistics, reference):
    if not statistics['count']:
        return None
    mean, variance = np.asarray(statistics['mean']), np.asarray(statistics['var'])
    configured_mean, configured_var = np.asarray(reference['effective_mean']), np.asarray(reference['var'])
    return dict(mean_delta=(mean - configured_mean).tolist(),
                variance_ratio=(variance / configured_var).tolist(),
                normalized_mean=((mean - configured_mean) / np.sqrt(configured_var)).tolist(),
                normalized_std=np.sqrt(variance / configured_var).tolist())


def model_processor(config, seed, threads):
    feeder = config.get('train_feeder_args', {})
    if config.get('feeder', 'feeder.feeder_ntu.Feeder') != 'feeder.feeder_ntu.Feeder':
        raise ValueError('Model-input audit currently supports feeder.feeder_ntu.Feeder only')
    if any(feeder.get(name, False) for name in ('normalization', 'bone', 'vel')):
        raise ValueError('This audit supports the native unnormalized joint feeder only')
    window = feeder.get('window_size', config['model_args'].get('num_frames', 120))
    interval = feeder.get('p_interval', [.5, 1.])
    interval = [interval] if isinstance(interval, (int, float)) else list(interval)
    if (not isinstance(window, int) or window <= 0 or len(interval) not in (1, 2)
            or not all(np.isfinite(p) and 0 < p <= 1 for p in interval)
            or interval[0] > interval[-1]):
        raise ValueError('Invalid window_size/p_interval for native preprocessing')
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Model-input view needs CPU PyTorch; --view raw needs only NumPy') from exc
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from feeder import tools
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)

    def process(sample):
        # Exactly the feeder rule, including its sum-based valid-frame count.
        valid = int(np.sum(sample.sum(0).sum(-1).sum(-1) != 0))
        if not valid:
            raise ValueError('Native feeder finds zero valid frames')
        result = tools.valid_crop_resize(sample, valid, interval, window)
        if feeder.get('flip', False):
            result = tools.random_spatial_flip(result, p=.5, dataset='ntu')
        if feeder.get('random_rot', False):
            result = tools.random_rot(result)
        return result

    metadata = dict(window_size=window, p_interval=interval, flip=bool(feeder.get('flip', False)),
                    random_rot=bool(feeder.get('random_rot', False)), seed=seed,
                    native_functions='feeder.tools.valid_crop_resize/random_spatial_flip/random_rot',
                    ignored_source_augmentation=['source_rot', 'joint_noise', 'frame_noise',
                                                 'motion_noise', 'impulse_noise'],
                    diffusion_or_flow_noise=False)
    return process, metadata


def audit(config_path, args):
    config, identity = load_config(config_path)
    if config.get('train_feeder_args', {}).get('split', 'train') != 'train':
        raise ValueError('Only train split may be used to fit normalization')
    primary = config_reference(config, identity)
    references = [primary]
    paths = list(args.compare_configs or [])
    if not paths:
        paths = [str(Path(config_path).with_name(name + '_madiff.yaml'))
                 for name in ('linprobe', 'finetune')]
        paths = [path for path in paths if Path(path).is_file()]
    for path in paths:
        other, other_identity = load_config(path)
        if other_identity['path'] != identity['path']:
            references.append(config_reference(other, other_identity))
    data_path = args.data_path or config.get('train_feeder_args', {}).get('data_path')
    if not data_path:
        raise ValueError('Pass --data-path or configure train_feeder_args.data_path')
    process, transform = (model_processor(config, args.seed, args.threads)
                          if args.view != 'raw' else (None, None))
    modes = ('person0', 'all') if args.person_mode == 'compare' else (args.person_mode,)
    groups = {}
    counters = dict(scanned=0, skipped_all_zero=0, person0_empty=0,
                    native_frame_rule_differs=0, nonprefix_time_padding=0)
    started = time.perf_counter()
    with TrainArray(data_path) as source:
        if source.shape[0] == 0:
            raise ValueError('Training array is empty')
        for start, batch in source.batches(args.chunk_size, args.max_samples):
            for offset, stored in enumerate(batch):
                index = start + offset
                try:
                    sample = sample_ctvm(stored, args.joints, args.layout)
                    counters['scanned'] += 1
                    occupied = np.any(sample != 0, axis=(0, 2, 3))
                    if not occupied.any():
                        counters['skipped_all_zero'] += 1
                        continue
                    if not np.any(sample[..., 0] != 0):
                        counters['person0_empty'] += 1
                    native_frames = int(np.sum(sample.sum(0).sum(-1).sum(-1) != 0))
                    if native_frames != int(occupied.sum()):
                        counters['native_frame_rule_differs'] += 1
                    if not np.array_equal(occupied, np.arange(len(occupied)) < occupied.sum()):
                        counters['nonprefix_time_padding'] += 1
                    if args.view != 'model':
                        update_groups(groups, sample[:, occupied], 'raw_valid', modes)
                    if process is not None:
                        for _ in range(args.draws_per_sample):
                            processed = process(sample)
                            if not np.isfinite(processed).all():
                                raise ValueError('Native preprocessing produced nonfinite coordinates')
                            update_groups(groups, processed, 'model_input', modes)
                except (ValueError, RuntimeError) as exc:
                    raise ValueError('Training sample %d: %s' % (index, exc)) from exc
            if args.progress_every and counters['scanned'] % args.progress_every < len(batch):
                print('%s: %d/%d samples, %.1fs' % (
                    Path(config_path).parent.name, counters['scanned'], source.shape[0],
                    time.perf_counter() - started), flush=True)
        data_identity = source.metadata
    if not groups:
        raise ValueError('No nonempty training skeletons were found')
    results = []
    for (view, people, shifted, padding), moments in sorted(groups.items()):
        statistics = moments.result()
        usable = bool(statistics['count'] and all(value > 0 for value in statistics['var']))
        comparisons = []
        for reference in references:
            if reference['self_shift'] == shifted:
                comparisons.append(dict(config=reference['path'],
                    values=compare_statistics(statistics, reference)))
        results.append(dict(view=view, person_mode=people, self_shift=shifted, padding=padding,
            statistics=statistics, comparisons=comparisons,
            candidate_usable=usable,
            proposed_input_mean=([0., 0., 0.] if shifted else statistics['mean']) if usable else None,
            proposed_input_var=statistics['var'] if usable else None,
            configured_pt_person_mode=people == primary['person_mode']))
    differences = [ref['path'] for ref in references[1:] if any(
        ref[name] != primary[name] for name in ('effective_mean', 'var', 'self_shift'))]
    return dict(schema=SCHEMA, source=data_identity, config=identity, references=references,
        normalization_mismatch_configs=differences, counters=counters, results=results,
        protocol=dict(train_only=True, requested_view=args.view, raw_time_rule='any nonzero coordinate',
            model_preprocessing=transform, person_mode=args.person_mode,
            self_shift_variants=[False, True], variance_ddof=0,
            arithmetic='float64 moments/centering; native crop/resize/rotation uses CPU Torch float32',
            mask_rule='all-zero person-frame over XYZ and all joints, before centering',
            chunk_size=args.chunk_size, draws_per_sample=args.draws_per_sample,
            max_samples=args.max_samples, complete_train_scan=counters['scanned'] == data_identity['shape'][0],
            reference_comparisons='Each config normalization is evaluated on the audited PT views; '
                                  'LP/FT crop pipelines are not rerun'), elapsed_seconds=time.perf_counter()-started)


def vector(values):
    return 'n/a' if values is None else '[' + ', '.join('%.7g' % value for value in values) + ']'


def print_report(report):
    primary = report['references'][0]
    view = 'model_input' if report['protocol']['requested_view'] != 'raw' else 'raw_valid'
    print('\n%s | configured PT=%s, self_shift=%s | %d training samples' % (
        Path(primary['path']).parent.name, primary['person_mode'], primary['self_shift'],
        report['counters']['scanned']))
    print('Existing effective mean=%s var=%s' % (vector(primary['effective_mean']), vector(primary['var'])))
    for item in report['results']:
        if item['view'] != view or item['self_shift'] != primary['self_shift']:
            continue
        stats = item['statistics']
        comparison = next(entry['values'] for entry in item['comparisons'] if entry['config'] == primary['path'])
        print('  %s / %s: count=%d mean=%s var=%s' % (
            item['person_mode'], item['padding'], stats['count'], vector(stats['mean']), vector(stats['var'])))
        if comparison is not None:
            print('    with existing constants: normalized_mean=%s normalized_std=%s' % (
                vector(comparison['normalized_mean']), vector(comparison['normalized_std'])))
    for path in report['normalization_mismatch_configs']:
        print('  PT normalization differs from: ' + path)
    print('  counters: ' + json.dumps(report['counters']))
    if not report['protocol']['complete_train_scan']:
        print('  PREVIEW ONLY: not a full-training-set estimate')


def write_reports(reports, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for report in reports:
        stem = Path(report['config']['path']).parent.name + '__' + Path(report['config']['path']).stem
        path = directory / (stem + '.json')
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    path = directory / 'comparison.csv'
    fields = ['config', 'view', 'person_mode', 'self_shift', 'padding', 'count', 'reference']
    for name in ('mean', 'var', 'std', 'normalized_mean', 'normalized_std', 'variance_ratio'):
        fields.extend(name + '_' + axis for axis in 'xyz')
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for item in report['results']:
                for comparison in item['comparisons']:
                    row = dict(config=report['config']['path'], reference=comparison['config'],
                               **{key: item[key] for key in ('view','person_mode','self_shift','padding')},
                               count=item['statistics']['count'])
                    for name in ('mean', 'var', 'std', 'normalized_mean', 'normalized_std', 'variance_ratio'):
                        values = (item['statistics'].get(name) if name in ('mean', 'var', 'std')
                                  else (comparison['values'] or {}).get(name))
                        for axis, value in zip('xyz', values or ['', '', '']):
                            row[name + '_' + axis] = value
                    writer.writerow(row)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', default=[DEFAULT_CONFIG])
    parser.add_argument('--compare-configs', nargs='+', help='Default: sibling LP/FT configs')
    parser.add_argument('--data-path', help='Override training data path; one primary config only')
    parser.add_argument('--output-dir', default='output_dir/normalization_audit')
    parser.add_argument('--view', choices=('raw', 'model', 'both'), default='both')
    parser.add_argument('--person-mode', choices=('compare', 'person0', 'all'), default='compare')
    parser.add_argument('--layout', choices=('auto', 'flat', 'ctvm', 'tmvc'), default='auto')
    parser.add_argument('--joints', type=int, default=25)
    parser.add_argument('--draws-per-sample', type=int, default=1,
                        help='Fixed-seed Monte Carlo draws of the native clean PT transform')
    parser.add_argument('--max-samples', type=int, default=0, help='First N samples, preview only; 0=all')
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--progress-every', type=int, default=2000)
    return parser


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.data_path and len(args.configs) != 1:
        parser.error('--data-path override requires exactly one --configs entry')
    if (min(args.joints, args.draws_per_sample, args.chunk_size, args.threads) <= 0
            or min(args.max_samples, args.progress_every, args.seed) < 0):
        parser.error('Counts must be positive; max-samples/progress-every/seed must be nonnegative')
    reports = []
    for path in args.configs:
        report = audit(path, args)
        reports.append(report)
        print_report(report)
        write_reports(reports, args.output_dir)  # retain completed datasets if a later one fails
    print('\nReports: ' + str(Path(args.output_dir).resolve()))


if __name__ == '__main__':
    main()
