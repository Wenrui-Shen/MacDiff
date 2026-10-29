"""Inspect two random person-0 global descriptions before/after a trained remap.

Default: reuse the exact frozen-CLIP, unit-RMS training cache. --clip-model
optionally re-encodes the same descriptions using the cache's local CLIP model.
Read-only, Python 3.8 compatible; no encoder/decoder construction or training.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import secrets

import numpy as np


def as_dict(value):
    if isinstance(value, dict):
        return value
    return vars(value)


def select_pair(person_valid, seed=None):
    valid = np.asarray(person_valid)
    if valid.ndim != 2 or valid.shape[1] != 2 or valid.dtype != np.bool_:
        raise ValueError('Expected boolean [samples,2] person validity')
    eligible = np.flatnonzero(valid[:, 0])
    if len(eligible) < 2:
        raise ValueError('Need at least two descriptions with a valid person0')
    seed = secrets.randbits(32) if seed is None else seed
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    ids = np.random.RandomState(seed).choice(eligible, 2, replace=False)
    return ids, seed, len(eligible)


def read_labels(data_path, count):
    # Access only y_train, never decompress the large x_train array.
    with np.load(data_path, allow_pickle=False) as data:
        labels = data['y_train']
    if (labels.ndim != 2 or labels.shape[0] != count or labels.shape[1] < 2
            or not np.isfinite(labels).all()
            or not np.all((labels > 0).sum(axis=1) == 1)):
        raise ValueError('Expected one positive class per row of y_train, aligned with cache')
    return labels.argmax(axis=1)


def read_descriptions(path, indices, count):
    records = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(records, list) or len(records) != count:
        raise ValueError('Cache samples.json must contain the complete train split')
    by_id = {}
    for record in records:
        index = record.get('sample_index')
        if type(index) is not int or not 0 <= index < count or index in by_id:
            raise ValueError('Invalid/duplicate sample_index in samples.json')
        by_id[index] = record
    texts = []
    for index in indices:
        persons = by_id[int(index)].get('persons', [])
        person0 = [p for p in persons if p.get('person_index') == 0]
        if len(person0) != 1:
            raise ValueError('Missing/duplicate person0 description: ' + str(index))
        text = person0[0].get('global')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Empty person0 global description: ' + str(index))
        texts.append(text.strip())
    return texts


def vector_summary(vector, show_dims=8):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError('Expected a finite nonempty feature vector')
    if type(show_dims) is not int or show_dims < 1:
        raise ValueError('show-dims must be positive')
    k = min(show_dims, len(vector))
    return dict(dim=len(vector), norm=float(np.linalg.norm(vector)),
                rms=float(np.sqrt(np.mean(vector ** 2))),
                head=vector[:k].tolist(), tail=vector[-k:].tolist(),
                head_abs=np.abs(vector[:k]).tolist(), tail_abs=np.abs(vector[-k:]).tolist(),
                tail_start=len(vector) - k)


def pair_metrics(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if (first.ndim != 1 or first.shape != second.shape or not len(first)
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError('Comparison requires two finite vectors with the same dimension')
    a, b = np.linalg.norm(first), np.linalg.norm(second)
    if min(a, b) < 1e-12:
        raise ValueError('Cannot compute cosine for a zero feature vector')
    delta = first - second
    return dict(cosine=float(np.clip(np.dot(first / a, second / b), -1., 1.)),
                l2=float(np.linalg.norm(delta)),
                unit_l2=float(np.linalg.norm(first / a - second / b)),
                mean_abs_diff=float(np.abs(delta).mean()),
                max_abs_diff=float(np.abs(delta).max()))


def validate_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict) or not {'model', 'args', 'epoch'} <= set(checkpoint):
        raise ValueError('Use a PT checkpoint containing model, args and epoch')
    saved = as_dict(checkpoint['args'])
    options = as_dict(saved.get('model_args', {}))
    if options.get('text_target_mode') not in ('sample_target_blend', 'ema_remap'):
        raise ValueError('Need a trained residual remap checkpoint; fixed-CLIP/native runs have no remap')
    state = checkpoint['model']
    if not isinstance(state, dict):
        raise ValueError('Checkpoint model must be a state dictionary')
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    weights = {k[len('text_remap.'):]: v for k, v in state.items() if k.startswith('text_remap.')}
    expected = {'body.0.weight', 'body.0.bias', 'body.2.weight', 'body.2.bias'}
    if set(weights) != expected:
        raise ValueError('Missing/incompatible online text_remap weights; not using EMA/target-bank substitutes')
    if not saved.get('text_cache_identity'):
        raise ValueError('PT checkpoint lacks text_cache_identity')
    return saved, options, weights


def load_checkpoint(path):
    import torch
    # Old Torch 1.8 has no weights_only keyword; current checkpoints include args.
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(str(path), **kwargs)


def check_cache_identity(manifest, saved_identity):
    if manifest.get('protocol') != 'macdiff_clip_sentence_cache_v3':
        raise ValueError('This diagnostic uses the current v3 global-sentence cache')
    identity = {k: manifest[k] for k in ('protocol', 'identity', 'files')}
    if identity != saved_identity:
        raise ValueError('Checkpoint/cache identity mismatch')
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def clip_weight_name(expected_files):
    names = [name for name in ('model.safetensors', 'pytorch_model.bin') if name in expected_files]
    if len(names) != 1:
        raise ValueError('Cache identity must record exactly one CLIP weight file')
    return names[0]


def encode_live(texts, model_path, manifest, cached, device):
    import torch
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizerFast
    from util.clip_text_cache import file_identity
    from util.structured_text_cache import unit_rms
    root = Path(model_path)
    expected_files = manifest['identity']['clip']
    weight_name = clip_weight_name(expected_files)
    for name, identity in expected_files.items():
        if file_identity(root / name) != identity:
            raise ValueError('Local CLIP differs from the training cache: ' + name)
    config = CLIPTextConfig.from_pretrained(root, local_files_only=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(root, local_files_only=True)
    model = CLIPTextModelWithProjection(config)
    if weight_name == 'model.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(root / 'model.safetensors'))
    else:
        state = load_checkpoint(root / weight_name)
    expected = model.state_dict()
    extra = {k for k in state if k.startswith(('text_model.', 'text_projection.'))} - set(expected)
    extra.discard('text_model.embeddings.position_ids')
    if set(expected) - set(state) or extra:
        raise ValueError('CLIP text checkpoint does not match its architecture')
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    del state, expected
    model.eval().requires_grad_(False).to(device)
    ids = tokenizer(texts, padding=False, truncation=False)['input_ids']
    if any(len(row) > config.max_position_embeddings for row in ids):
        raise ValueError('Description exceeds CLIP context; no silent truncation')
    batch = tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to(device)
    with torch.no_grad():
        raw = model(**batch).text_embeds.float().cpu().numpy()
    values = unit_rms(raw)
    if values.shape != cached.shape:
        raise ValueError('CLIP dimensions differ from the cache')
    comparisons = [pair_metrics(a, b) for a, b in zip(values, cached)]
    if any(row['cosine'] < .9999 for row in comparisons):
        raise ValueError('Re-encoded descriptions disagree with cached vectors: ' + str(comparisons))
    return values, raw, comparisons


def print_vector(name, vector, show_dims):
    item = vector_summary(vector, show_dims)
    fmt = lambda x: '[' + ', '.join('{:+.6f}'.format(v) for v in x) + ']'
    print('  {}: dim={} norm={:.9f} RMS={:.9f}'.format(name, item['dim'], item['norm'], item['rms']))
    print('    head[0:{}] values={} abs={}'.format(len(item['head']), fmt(item['head']), fmt(item['head_abs'])))
    print('    tail[{}:{}] values={} abs={}'.format(item['tail_start'], item['dim'], fmt(item['tail']), fmt(item['tail_abs'])))


def print_metrics(name, metrics):
    print('{}: cosine={:.9f}, L2={:.9f}, unit_L2={:.9f}, mean_abs_diff={:.9f}, max_abs_diff={:.9f}'.format(
        name, metrics['cosine'], metrics['l2'], metrics['unit_l2'],
        metrics['mean_abs_diff'], metrics['max_abs_diff']))


def run(args):
    import torch
    from model.transformer_macdiff_text import ResidualTextRemap
    from util.person_text_cache import load_person_token_cache
    if args.show_dims < 1:
        raise ValueError('show-dims must be positive')
    checkpoint = load_checkpoint(args.checkpoint)
    saved, options, weights = validate_checkpoint(checkpoint)
    epoch = checkpoint['epoch']
    dim = int(options.get('text_input_dim', 512))
    hidden = int(options.get('text_hidden_dim', 512))
    remap = ResidualTextRemap(dim, hidden)
    remap.load_state_dict(weights, strict=True)
    remap.eval().requires_grad_(False).to(args.device)
    del weights, checkpoint
    cache_path = args.text_cache or saved.get('text_cache')
    data_path = args.data_path or saved.get('train_feeder_args', {}).get('data_path')
    if not cache_path or not data_path:
        raise ValueError('Pass --text-cache and --data-path or use a PT checkpoint with those saved paths')
    manifest = json.loads((Path(cache_path) / 'manifest.json').read_text(encoding='utf-8'))
    identity_hash = check_cache_identity(manifest, saved['text_cache_identity'])
    labels = read_labels(data_path, manifest['sample_count'])
    print('Validating cache: {} ...'.format('full checksums/rows' if args.full_validation else 'headers/sizes + saved identity'), flush=True)
    cache = load_person_token_cache(cache_path, data_path, len(labels),
                                    skip_full_validation=not args.full_validation)
    try:
        valid = np.load(Path(cache_path) / 'person_valid.npy', allow_pickle=False)
        indices, seed, eligible = select_pair(valid, args.seed)
        texts = read_descriptions(Path(cache_path) / 'samples.json', indices, len(labels))
        clip = cache.get_global_batch(indices, one_person=True)['text_features'].copy()
    finally:
        for values in (cache.global_text, cache.features, cache.mask):
            values._mmap.close()
    if clip.shape != (2, dim) or not np.isfinite(clip).all():
        raise ValueError('Invalid selected CLIP features')
    if not np.allclose(np.mean(clip ** 2, axis=1), 1., atol=2e-5):
        raise ValueError('Expected the training unit-RMS CLIP features')
    raw_clip, reencoded = None, None
    if args.clip_model:
        clip, raw_clip, reencoded = encode_live(texts, args.clip_model, manifest, clip, args.device)
    with torch.no_grad():
        mapped = remap(torch.from_numpy(clip).to(args.device)).float().cpu().numpy()
    print('checkpoint={} epoch={} model={}'.format(args.checkpoint, epoch, saved.get('model')))
    print('source={} person=0 scope=global_sentence remap=online_checkpoint (before target smoothing)'.format(
        'live_frozen_CLIP' if args.clip_model else 'frozen_CLIP_training_cache'))
    print('seed={} eligible_person0_samples={} same_class={} cache_identity_sha256={}'.format(
        seed, eligible, bool(labels[indices[0]] == labels[indices[1]]), identity_hash))
    print('Feature previews include signed values and absolute values; cosine uses full SIGNED vectors.')
    for row, index in enumerate(indices):
        print('\nSample {}: sample_id=train_{} sample_index={} class_id_0based={} class_id_1based={} person_id=0'.format(
            row + 1, index, index, labels[index], labels[index] + 1))
        print('  description=' + json.dumps(texts[row], ensure_ascii=False))
        if raw_clip is not None:
            print_vector('CLIP_projected_raw', raw_clip[row], args.show_dims)
            print_metrics('  live_CLIP versus cache', reencoded[row])
        print_vector('CLIP_unit_RMS (remap input)', clip[row], args.show_dims)
        print_vector('remap_unit_RMS', mapped[row], args.show_dims)
        print_metrics('  same sample CLIP versus remap', pair_metrics(clip[row], mapped[row]))
    before, after = pair_metrics(*clip), pair_metrics(*mapped)
    print('\nBetween the TWO DIFFERENT SAMPLES:')
    print_metrics('  CLIP', before)
    print_metrics('  remap', after)
    print('  cosine_change(remap-CLIP)={:+.9f}'.format(after['cosine'] - before['cosine']))
    print('Two samples show one pair, not a population collapse/semantic-quality verdict. No diagnostic output files were written.')


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='PT checkpoint with a learned residual text_remap')
    parser.add_argument('--text-cache', default=None)
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--device', default='cpu', help='Only two remap inputs; CPU is sufficient')
    parser.add_argument('--seed', type=int, default=None, help='Default: fresh OS-random seed on every run')
    parser.add_argument('--show-dims', type=int, default=8)
    parser.add_argument('--full-validation', action='store_true', help='Also rehash data/cache and validate all rows')
    parser.add_argument('--clip-model', default=None, help='Optional local frozen CLIP path: re-encode instead of reusing vectors')
    return parser


if __name__ == '__main__':
    run(get_parser().parse_args())
