"""Frozen Stage1 versus caption/CLIP geometry on identical balanced batches.

Plan/metrics require NumPy only. Extraction uses the existing MacDiff and
Transformers environments; neither encoder is trained by this experiment.
"""
import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

from stage1_readout import fingerprint

PROTOCOL = 'stage1_caption_balanced_geometry_v1'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def normalize(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms < 1e-10):
        raise ValueError('Non-finite or zero feature vector')
    return features / norms


def balanced_batches(indices, labels, count=100, classes_per_batch=16, per_class=8, seed=42):
    indices, labels = np.asarray(indices), np.asarray(labels)
    classes = np.unique(labels)
    if count < 1 or classes_per_batch < 2 or per_class < 2 or len(classes) < classes_per_batch:
        raise ValueError('Need positive batch count, >=2 classes and >=2 samples/class')
    buckets = {int(c): indices[labels == c] for c in classes}
    if len(np.unique(indices)) != len(indices) or any(len(v) < per_class for v in buckets.values()):
        raise ValueError('Need unique sample IDs and enough samples in every class')
    rng = np.random.RandomState(seed)
    result = []
    for _ in range(count):
        selected = rng.choice(classes, classes_per_batch, replace=False)
        batch = np.concatenate([rng.choice(buckets[int(c)], per_class, replace=False) for c in selected])
        rng.shuffle(batch)
        result.append(batch.tolist())
    return result


def read_captions(paths, sample_count):
    rows, versions = {}, set()
    counts = {'accepted_records': 0, 'other_records': 0, 'duplicate_accepted_records': 0}
    for path in paths:
        with open(path, encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f'{path}:{line_number}: invalid/incomplete JSON; retry after writing finishes') from exc
                if row.get('status') != 'accepted':
                    counts['other_records'] += 1
                    continue
                index = int(row['sample_index'])
                if row.get('source_split') != 'train' or not 0 <= index < sample_count:
                    raise ValueError('Caption split/index does not match x_train: ' + str(index))
                version = (row.get('model'), row.get('prompt', {}).get('sha256'))
                if not all(version):
                    raise ValueError('Missing caption model/prompt hash')
                versions.add(version)
                persons = sorted(row['texts'], key=lambda item: item['person_index'])
                if (len(persons) not in (1, 2) or len(persons) != row['actor_count']
                        or [p['person_index'] for p in persons] != list(range(len(persons)))):
                    raise ValueError('Invalid person alignment: ' + str(index))
                texts = [p['text'].strip() for p in persons]
                if any(not text for text in texts):
                    raise ValueError('Empty accepted text: ' + str(index))
                counts['accepted_records'] += 1
                counts['duplicate_accepted_records'] += int(index in rows)
                rows[index] = texts  # last accepted record in supplied file order
    if len(versions) != 1:
        raise ValueError('Use exactly one caption model and prompt hash; found ' + str(versions))
    return rows, sorted(versions), counts


def make_plan(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError('Use a new output directory; plans are immutable')
    with np.load(args.data_path, allow_pickle=False) as archive:
        one_hot = archive['y_train']
    if (one_hot.ndim != 2 or one_hot.shape[1] != 60
            or not np.all((one_hot > 0).sum(axis=1) == 1)):
        raise ValueError('Expected NTU60 one-hot y_train')
    labels = one_hot.argmax(axis=1)
    captions, versions, counts = read_captions(args.captions, len(labels))
    ids = np.array(sorted(captions), dtype=np.int64)
    coverage = [{ 'class_id': c, 'total': int((labels == c).sum()),
                  'accepted': int((labels[ids] == c).sum())} for c in range(60)]
    if any(row['accepted'] < args.samples_per_class for row in coverage):
        raise ValueError('Every one of the 60 classes needs enough accepted captions: ' + str(coverage))
    batches = balanced_batches(ids, labels[ids], args.num_batches, args.classes_per_batch,
                               args.samples_per_class, args.seed)
    union = sorted({i for batch in batches for i in batch})
    plan = {
        'protocol': PROTOCOL, 'data': fingerprint(args.data_path),
        'caption_sources': [fingerprint(p) for p in args.captions],
        'caption_version': versions, 'caption_counts': counts,
        'duplicate_policy': 'last accepted in supplied file/line order',
        'coverage': coverage, 'seed': args.seed, 'batches': batches,
        'classes_per_batch': args.classes_per_batch, 'samples_per_class': args.samples_per_class,
        'samples': [{'index': i, 'label': int(labels[i]), 'texts': captions[i]} for i in union],
    }
    output.mkdir(parents=True)
    write_json(output / 'plan.json', plan)
    print(f'Plan: {len(batches)} batches x {args.classes_per_batch * args.samples_per_class}; '
          f'{len(union)} unique samples. Saved {output / "plan.json"}', flush=True)


def row_metrics(similarity, labels):
    """Anchor AUC (ties=1/2), tie-averaged P@k and anchor-balanced cosine gap."""
    labels = np.asarray(labels)
    result = {key: [] for key in ('auc', 'p1', 'p5', 'same_cosine', 'different_cosine', 'gap')}
    for i in range(len(labels)):
        valid = np.arange(len(labels)) != i
        scores = similarity[i, valid]
        positive = labels[valid] == labels[i]
        pos, neg = scores[positive], scores[~positive]
        if not len(pos) or not len(neg):
            raise ValueError('Every anchor needs both same- and different-class partners')
        delta = pos[:, None] - neg[None, :]
        result['auc'].append(float(((delta > 0) + 0.5 * (delta == 0)).mean()))
        for k, name in ((1, 'p1'), (5, 'p5')):
            k = min(k, len(scores))
            threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
            above, tied = scores > threshold, scores == threshold
            hits = positive[above].sum() + (k - above.sum()) * positive[tied].mean()
            result[name].append(float(hits / k))
        result['same_cosine'].append(float(pos.mean()))
        result['different_cosine'].append(float(neg.mean()))
        result['gap'].append(float(pos.mean() - neg.mean()))
    return {key: np.asarray(value) for key, value in result.items()}


def summarize(features, plan):
    features = normalize(features)
    samples = plan['samples']
    lookup = {row['index']: i for i, row in enumerate(samples)}
    labels = np.array([row['label'] for row in samples])
    per_batch, per_class = [], {}
    for batch in plan['batches']:
        ids = np.array([lookup[i] for i in batch])
        current_labels = labels[ids]
        metrics = row_metrics(features[ids] @ features[ids].T, current_labels)
        class_rows = []
        for c in np.unique(current_labels):
            row = {k: float(v[current_labels == c].mean()) for k, v in metrics.items()}
            per_class.setdefault(int(c), []).append(row)
            class_rows.append(row)
        per_batch.append({k: float(np.mean([r[k] for r in class_rows])) for k in metrics})
    classes = {str(c): {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
               for c, rows in per_class.items()}
    macro = {k: float(np.mean([r[k] for r in classes.values()])) for k in per_batch[0]}
    return {'class_macro': macro, 'per_class': classes, 'per_batch': per_batch,
            'batch_std': {k: float(np.std([r[k] for r in per_batch])) for k in per_batch[0]},
            'note': 'Repeated batches share samples; batch SD is descriptive, not an independent-sample CI.'}


def cached_features(root, name, identity, producer):
    path, metadata = root / (name + '.npy'), root / (name + '.json')
    if path.exists() and metadata.exists():
        if json.loads(metadata.read_text(encoding='utf-8')) != identity:
            raise ValueError('Cache provenance changed: ' + str(path))
        print('Reusing ' + str(path), flush=True)
        return np.load(path, allow_pickle=False)
    values = normalize(producer())
    temporary = root / (name + '.tmp.npy')
    np.save(temporary, values)
    temporary.replace(path)
    write_json(metadata, identity)
    return values


def extract_text(plan, model_path, batch_size, device):
    import torch
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizerFast
    root = Path(model_path)
    config = CLIPTextConfig.from_pretrained(root, local_files_only=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(root, local_files_only=True)
    model = CLIPTextModelWithProjection(config)
    if (root / 'model.safetensors').exists():
        from safetensors.torch import load_file
        state = load_file(str(root / 'model.safetensors'))
    else:
        # Official HF CLIP checkpoint is a tensor state dict, with text and vision.
        # Instantiate the text architecture directly: no auto-model dependencies.
        state = torch.load(root / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected_text = {k for k in state if k.startswith(('text_model.', 'text_projection.'))} - set(expected)
    unexpected_text.discard('text_model.embeddings.position_ids')  # legacy persistent buffer
    if missing or unexpected_text:
        raise ValueError(f'CLIP text checkpoint mismatch: {missing}, {unexpected_text}')
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    del state, expected
    model.eval().requires_grad_(False).to(device)
    texts = [text for sample in plan['samples'] for text in sample['texts']]
    tokens = tokenizer(texts, padding=False, truncation=False)['input_ids']
    if any(len(row) > config.max_position_embeddings for row in tokens):
        raise ValueError('Caption exceeds CLIP token limit; no silent truncation is allowed')
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True,
                              truncation=False, return_tensors='pt').to(device)
            vectors.append(model(**batch).text_embeds.float().cpu().numpy())
    vectors = normalize(np.concatenate(vectors))
    result, offset = [], 0
    for sample in plan['samples']:
        count = len(sample['texts'])
        result.append(vectors[offset:offset + count].mean(axis=0))
        offset += count
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.stack(result)


def extract_skeleton(plan, checkpoint, config_path, batch_size, device):
    import torch
    import yaml
    from compare_stage1_readouts import load_encoder
    from feeder.feeder_stage2 import FeederStage2
    from model.transformer_downstream import Transformer
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    model_args = config['model_args']
    if (model_args.get('protocol') != 'linprobe2' or model_args.get('dim_feat') != 256
            or model_args.get('num_joints') != 25 or model_args.get('num_frames') != 120):
        raise ValueError('Expected native NTU60 linprobe2 6400-D configuration')
    model = Transformer(**model_args)
    model.head.fc = torch.nn.Identity()
    print('Strict encoder transfer: ' + str(load_encoder(model, checkpoint)), flush=True)
    model.eval().requires_grad_(False).to(device)
    dataset = FeederStage2(plan['data']['path'], base_p_interval=(0.95,), window_size=120)
    ids = [row['index'] for row in plan['samples']]
    if [int(dataset.label[i]) for i in ids] != [row['label'] for row in plan['samples']]:
        raise ValueError('Data labels changed relative to plan')
    result = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch = torch.from_numpy(np.stack([dataset.get_base_sample(i)
                                              for i in ids[start:start + batch_size]])).to(device)
            values = model(batch).float().cpu().numpy()
            if values.shape[1] != 6400:
                raise ValueError('Expected 6400-dimensional skeleton features')
            result.append(values)
            if start == 0 or (start // batch_size) % 100 == 0:
                print(f'  skeleton {min(start + batch_size, len(ids))}/{len(ids)}', flush=True)
    del model, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(result)


def run(args):
    root = Path(args.output_dir)
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    if plan['protocol'] != PROTOCOL or fingerprint(plan['data']['path']) != plan['data']:
        raise ValueError('Protocol or data file changed since planning')
    checkpoints = sorted({p for pattern in args.checkpoints for p in glob.glob(pattern)},
                         key=lambda p: int(Path(p).stem.split('-')[-1]))
    if not checkpoints or any(not glob.glob(pattern) for pattern in args.checkpoints):
        raise FileNotFoundError('One or more checkpoint paths/patterns matched no files')
    import torch
    torch.manual_seed(plan['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    clip_root = Path(args.clip_model)
    clip_files = [fingerprint(p) for p in sorted(clip_root.iterdir())
                  if p.is_file() and p.suffix in ('.json', '.txt', '.bin', '.safetensors')]
    common = {'plan_sha256': digest(plan), 'protocol': PROTOCOL,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'torch': torch.__version__, 'device': args.device}
    import transformers
    print('Extracting/reusing CLIP text features...', flush=True)
    text = cached_features(root, 'text_features', {
        **common, 'clip_files': clip_files, 'transformers': transformers.__version__,
        'pooling': 'L2 each person, average persons, L2 sample'},
        lambda: extract_text(plan, args.clip_model, args.text_batch_size, args.device))
    text_stats = summarize(text, plan)
    write_json(root / 'text_metrics.json', text_stats)
    all_results = []
    for checkpoint in checkpoints:
        identity = {**common, 'checkpoint': fingerprint(checkpoint), 'config': fingerprint(args.config),
                    'pooling': 'linprobe2 fc=Identity eval; full tokens; both persons; center95 resize120'}
        name = Path(checkpoint).stem + '_' + digest(identity)[:10]
        print('Extracting/reusing ' + checkpoint, flush=True)
        skeleton = cached_features(root, name, identity,
            lambda: extract_skeleton(plan, checkpoint, args.config, args.micro_batch_size, args.device))
        stats = summarize(skeleton, plan)
        delta = {k: text_stats['class_macro'][k] - stats['class_macro'][k]
                 for k in text_stats['class_macro']}
        result = {'checkpoint': fingerprint(checkpoint), 'epoch_index': int(Path(checkpoint).stem.split('-')[-1]),
                  'skeleton': stats, 'text_minus_skeleton': delta,
                  'paired_batch_auc_differences': [t['auc'] - s['auc'] for t, s in
                                                 zip(text_stats['per_batch'], stats['per_batch'])]}
        write_json(root / (name + '_metrics.json'), result)
        all_results.append(result)
        write_json(root / 'summary.json', {'protocol': PROTOCOL, 'text': text_stats, 'checkpoints': all_results})
        print(f"{Path(checkpoint).name}: AUC text={text_stats['class_macro']['auc']:.4f}, "
              f"skeleton={stats['class_macro']['auc']:.4f}, delta={delta['auc']:+.4f}", flush=True)
    export_report(root, text_stats, all_results)


def export_report(root, text_stats, results):
    keys = ('auc', 'p1', 'p5', 'same_cosine', 'different_cosine', 'gap')
    with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['representation', 'epoch_index', *keys])
        writer.writeheader()
        writer.writerow({'representation': 'CLIP text', 'epoch_index': '', **text_stats['class_macro']})
        for result in results:
            writer.writerow({'representation': 'Stage1', 'epoch_index': result['epoch_index'],
                             **result['skeleton']['class_macro']})
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib unavailable; JSON and CSV reports are complete.', flush=True)
        return
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    epochs = [r['epoch_index'] + 1 for r in results]
    for ax, key, title in zip(axes, ('auc', 'p1', 'p5'), ('Anchor AUC', 'Precision@1', 'Precision@5')):
        ax.plot(epochs, [r['skeleton']['class_macro'][key] for r in results], 'o-', label='Stage1 skeleton')
        ax.axhline(text_stats['class_macro'][key], color='tab:orange', linestyle='--', label='CLIP text')
        ax.set(xlabel='Completed training epochs', ylabel=title, ylim=(0, 1))
        ax.grid(alpha=0.2)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(root / 'geometry_curve.png', dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='Freeze class-balanced sample IDs/texts without loading models')
    plan.add_argument('--data-path', required=True)
    plan.add_argument('--captions', nargs='+', required=True)
    plan.add_argument('--output-dir', required=True)
    plan.add_argument('--num-batches', type=int, default=100)
    plan.add_argument('--classes-per-batch', type=int, default=16)
    plan.add_argument('--samples-per-class', type=int, default=8)
    plan.add_argument('--seed', type=int, default=42)
    run_parser = sub.add_parser('run', help='Extract/cache features, then compare all fixed batches')
    run_parser.add_argument('--output-dir', required=True)
    run_parser.add_argument('--checkpoints', nargs='+', required=True)
    run_parser.add_argument('--clip-model', required=True)
    run_parser.add_argument('--config', default='config/ntu60_xsub_joint/linprobe_madiff.yaml')
    run_parser.add_argument('--micro-batch-size', type=int, default=4)
    run_parser.add_argument('--text-batch-size', type=int, default=64)
    run_parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.command == 'plan':
        make_plan(args)
    else:
        if args.micro_batch_size < 1 or args.text_batch_size < 1:
            parser.error('Extraction batch sizes must be positive')
        run(args)


if __name__ == '__main__':
    main()
