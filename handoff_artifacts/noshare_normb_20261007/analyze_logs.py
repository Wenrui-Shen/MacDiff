"""Reproduce the 2026-10-07 log summaries using only the Python standard library."""
import hashlib
import json
import math
import statistics
from pathlib import Path


ATTACHMENTS = Path('C:/Users/97537/.codex/attachments')
SOURCES = {
    'noshare_pt': '02d00a0e-de94-42cb-98f4-6ad1cc73af47',
    'noshare_lp': 'cc1abd5f-c4d1-481b-8670-badf284c8140',
    'normb_pt': '826649bf-4dae-4569-b341-3e1581c0cf92',
    'normb_lp': '4cb61fd1-2294-42bf-82dd-a231581f3661',
    'shared512_pt': '64857d73-03e2-4bf9-97ed-e6ca4fba0355',
}


def load_log(identifier, expected):
    path = ATTACHMENTS / identifier / '\u5df2\u7c98\u8d34\u7684\u6587\u672c.txt'
    raw = path.read_bytes()
    rows = []
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError('{}:{}: {}'.format(path, number, error))
            if not isinstance(row, dict):
                raise ValueError('Expected JSON object at {}:{}'.format(path, number))
            rows.append(row)
    epochs = [row['epoch'] for row in rows]
    nonfinite = [(row['epoch'], key) for row in rows for key, value in row.items()
                 if isinstance(value, (int, float)) and not math.isfinite(value)]
    integrity = {
        'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
        'count': len(rows), 'expected_count': expected,
        'epochs_in_order': epochs == list(range(expected)),
        'duplicate_epochs': sorted(epoch for epoch in set(epochs) if epochs.count(epoch) > 1),
        'missing_epochs': sorted(set(range(expected)) - set(epochs)),
        'nonfinite_values': nonfinite,
        'keys': sorted(set().union(*(row.keys() for row in rows))),
    }
    assert integrity['epochs_in_order'] and not nonfinite, integrity
    return rows, integrity


def distribution(rows, key):
    values = [row[key] for row in rows if key in row]
    return {'count': len(values), 'mean': statistics.mean(values),
            'population_std': statistics.pstdev(values), 'min': min(values), 'max': max(values)}


def window(rows, start, end, keys):
    selected = [row for row in rows if start <= row['epoch'] <= end]
    return {'start': start, 'end': end,
            'metrics': {key: distribution(selected, key) for key in keys}}


def lp_summary(rows):
    best = max(rows, key=lambda row: row['test_acc1'])
    best5 = max(rows, key=lambda row: row['test_acc5'])
    min_ce = min(rows, key=lambda row: row['test_loss'])
    max_ce = max(rows, key=lambda row: row['test_loss'])
    return {
        'best': best, 'last': rows[-1], 'first': rows[0], 'best_top5': best5,
        'min_test_ce': {'epoch': min_ce['epoch'], 'value': min_ce['test_loss']},
        'max_test_ce': {'epoch': max_ce['epoch'], 'value': max_ce['test_loss']},
        'last10': window(rows, 90, 99, ('test_acc1', 'test_acc5', 'train_loss', 'test_loss')),
        'last20': window(rows, 80, 99, ('test_acc1', 'test_acc5', 'train_loss', 'test_loss')),
        'stages': [window(rows, a, b, ('test_acc1', 'train_loss', 'test_loss'))
                   for a, b in ((0, 9), (10, 19), (20, 39), (40, 59), (60, 79), (80, 99))],
        'zero_lr_epochs': [row['epoch'] for row in rows if row['train_lr'] == 0],
        'n_parameters': sorted({row['n_parameters'] for row in rows}),
    }


def pt_summary(rows):
    metric_keys = [key for key in rows[0] if key.startswith('train_loss')]
    summary = {
        'first': rows[0], 'last': rows[-1],
        'minimum': {key: {'epoch': min(rows, key=lambda row: row[key])['epoch'],
                          'value': min(row[key] for row in rows)} for key in metric_keys},
        'last50': window(rows, 350, 399, metric_keys),
        'stages': [window(rows, a, b, metric_keys) for a, b in
                   ((0, 19), (20, 49), (50, 99), (100, 149), (150, 199),
                    (200, 249), (250, 299), (300, 349), (350, 399))],
        'cuda_peak_allocated_mb': max(row['train_cuda_peak_allocated_mb'] for row in rows),
        'cuda_peak_reserved_mb': max(row['train_cuda_peak_reserved_mb'] for row in rows),
    }
    if 'train_loss_skeleton_to_text' in rows[0]:
        residuals = [row['train_loss'] - (
            row['train_loss_diff'] + .02 * row['train_loss_uniformity']
            + .02 * row['train_loss_text_uniformity']
            + row['train_loss_text_to_skeleton'] + .1 * row['train_loss_skeleton_to_text'])
            for row in rows]
        summary['loss_formula_max_abs_error'] = max(abs(value) for value in residuals)
        summary['valid_local_tokens'] = sorted({row['train_text_valid_tokens'] for row in rows})
        summary['target_bank_counts'] = sorted({row['train_target_bank_initialized_samples'] for row in rows})
        for key in ('train_epoch_seconds', 'train_text_energy', 'train_text_batch_variance'):
            summary[key] = distribution(rows, key)
            summary[key + '_last50'] = distribution(rows[-50:], key)
        summary['total_epoch_hours'] = sum(row['train_epoch_seconds'] for row in rows) / 3600
        summary['median_epoch_seconds'] = statistics.median(row['train_epoch_seconds'] for row in rows)
        summary['s2t_last_over_min'] = (
            rows[-1]['train_loss_skeleton_to_text']
            / summary['minimum']['train_loss_skeleton_to_text']['value'])
    return summary


def main():
    loaded, integrity = {}, {}
    for name, identifier in SOURCES.items():
        loaded[name], integrity[name] = load_log(identifier, 100 if name.endswith('_lp') else 400)
    result = {
        'date': '2026-10-07', 'integrity': integrity,
        'statistics_note': 'Epoch SD describes within-run fluctuations, not independent-run uncertainty.',
        'protocol_note': 'Batch and configuration identities come from the user and supplied commands; logs do not include checkpoint args or server Git SHA.',
        'noshare_lp': lp_summary(loaded['noshare_lp']),
        'normb_lp': lp_summary(loaded['normb_lp']),
        'noshare_pt': pt_summary(loaded['noshare_pt']),
        'normb_pt': pt_summary(loaded['normb_pt']),
        'shared512_pt': pt_summary(loaded['shared512_pt']),
    }
    result['comparison'] = {
        'noshare_best_minus_reported_shared85_79_pp': result['noshare_lp']['best']['test_acc1'] - 85.79,
        'noshare_best_minus_normb_best_pp': result['noshare_lp']['best']['test_acc1'] - result['normb_lp']['best']['test_acc1'],
        'noshare_last20_minus_normb_last20_pp': (
            result['noshare_lp']['last20']['metrics']['test_acc1']['mean']
            - result['normb_lp']['last20']['metrics']['test_acc1']['mean']),
        'noshare_vs_shared_max_lr_difference': max(
            abs(a['train_lr'] - b['train_lr'])
            for a, b in zip(loaded['noshare_pt'], loaded['shared512_pt'])),
        'last50_loss_relative_changes_percent': {
            key: 100 * (result['noshare_pt']['last50']['metrics'][key]['mean']
                        / result['shared512_pt']['last50']['metrics'][key]['mean'] - 1)
            for key in result['noshare_pt']['last50']['metrics']},
    }
    destination = Path(__file__).with_name('summary.json')
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    compact = {'integrity': {name: {'count': value['count'], 'complete': value['epochs_in_order']}
                             for name, value in integrity.items()}, 'comparison': result['comparison']}
    for name in ('noshare_lp', 'normb_lp'):
        compact[name] = {key: result[name][key] for key in ('best', 'last', 'last10', 'last20')}
    for name in ('noshare_pt', 'normb_pt', 'shared512_pt'):
        compact[name] = {key: result[name][key] for key in ('minimum', 'last50')}
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
