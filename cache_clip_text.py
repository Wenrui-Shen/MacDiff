"""Cache all accepted NTU training captions with a frozen local HF CLIP encoder."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_stage1_text_geometry import read_captions
from util.clip_text_cache import (PROTOCOL, file_identity, global_features,
                                  load_cache, train_shape, write_json)


def build_plan(data_path, caption_paths):
    count = train_shape(data_path)[0]
    captions, versions, counts = read_captions(caption_paths, count)
    missing = sorted(set(range(count)) - set(captions))
    if missing:
        raise ValueError('Missing accepted captions: {} (first 20: {})'.format(
            len(missing), missing[:20]))
    samples = [{'sample_index': i, 'texts': captions[i]} for i in range(count)]
    return samples, versions, counts


def load_encoder(model_path, device):
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
        state = torch.load(root / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    expected = model.state_dict()
    unexpected = {k for k in state if k.startswith(('text_model.', 'text_projection.'))} - set(expected)
    unexpected.discard('text_model.embeddings.position_ids')
    if set(expected) - set(state) or unexpected:
        raise ValueError('CLIP text checkpoint keys do not match the architecture')
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, tokenizer, config


def run(args):
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    samples, versions, counts = build_plan(args.data_path, args.captions)
    root, clip = Path(args.output_dir), Path(args.clip_model)
    weight = clip / ('model.safetensors' if (clip / 'model.safetensors').exists()
                     else 'pytorch_model.bin')
    model_files = sorted(set(clip.glob('*.json')) | set(clip.glob('*.txt')) | {weight})
    print('Checking dataset, captions and CLIP fingerprints...', flush=True)
    identity = {
        'data': file_identity(args.data_path),
        'captions': [file_identity(p) for p in args.captions],
        'clip': {p.name: file_identity(p) for p in model_files},
        'caption_version': [list(v) for v in versions],
        'implementation': {p: file_identity(Path(__file__).parent / p) for p in (
            'cache_clip_text.py', 'util/clip_text_cache.py', 'compare_stage1_text_geometry.py')},
    }
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') != PROTOCOL or manifest['identity'] != identity:
            raise ValueError('Cache provenance changed; use a new output directory')
        if manifest['complete']:
            load_cache(root, expected_count=len(samples))
            print('Reusing complete cache: ' + str(root), flush=True)
            return
        if not args.resume:
            raise ValueError('Incomplete cache exists; pass --resume to continue')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Output directory is nonempty without a cache manifest')
    else:
        manifest = None

    # A complete cache can be reused without importing Torch/Transformers.
    import torch
    model, tokenizer, config = load_encoder(clip, args.device)
    pairs = [(s['sample_index'], person, text)
             for s in samples for person, text in enumerate(s['texts'])]
    texts = [row[2] for row in pairs]
    lengths = [len(ids) for ids in tokenizer(texts, padding=False, truncation=False)['input_ids']]
    too_long = [(pairs[i][:2], n) for i, n in enumerate(lengths)
                if n > config.max_position_embeddings]
    if too_long:
        raise ValueError('Captions exceed CLIP token limit (no truncation): ' + str(too_long[:20]))
    dim, count = config.projection_dim, len(samples)
    valid = np.zeros((count, 2), dtype=bool)
    for i, person, _ in pairs:
        valid[i, person] = True
    if manifest is None:
        root.mkdir(parents=True, exist_ok=True)
        values = np.lib.format.open_memmap(root / 'person_features.npy', mode='w+',
                                          dtype=np.float32, shape=(count, 2, dim))
        values[:] = 0
        values.flush()
        np.save(root / 'person_valid.npy', valid, allow_pickle=False)
        write_json(root / 'samples.json', samples)
        manifest = {'protocol': PROTOCOL, 'identity': identity, 'complete': False,
                    'sample_count': count, 'feature_dim': dim, 'completed_persons': 0,
                    'total_persons': len(pairs), 'caption_counts': counts,
                    'representation': 'projected text_embeds; FP32; per-person L2',
                    'duplicate_policy': 'last accepted; invalid/error records ignored'}
        write_json(manifest_path, manifest)
    else:
        values = np.load(root / 'person_features.npy', mmap_mode='r+', allow_pickle=False)
        if (values.shape != (count, 2, dim) or values.dtype != np.float32
                or not np.array_equal(np.load(root / 'person_valid.npy'), valid)
                or manifest['total_persons'] != len(pairs)
                or not 0 <= manifest['completed_persons'] <= len(pairs)):
            raise ValueError('Incomplete cache shape/progress does not match captions')
        done = pairs[:manifest['completed_persons']]
        if done:
            saved = values[[p[0] for p in done], [p[1] for p in done]]
            if not np.isfinite(saved).all() or not np.allclose(np.linalg.norm(saved, axis=1), 1, atol=1e-4):
                raise ValueError('Completed cache rows are corrupt')
    with torch.inference_mode():
        for start in range(manifest['completed_persons'], len(pairs), args.batch_size):
            end = min(start + args.batch_size, len(pairs))
            batch = tokenizer(texts[start:end], padding=True, truncation=False,
                              return_tensors='pt').to(args.device)
            vectors = model(**batch).text_embeds.float()
            norms = vectors.norm(dim=-1, keepdim=True)
            if not torch.isfinite(vectors).all() or (norms < 1e-8).any():
                raise ValueError('Invalid CLIP feature')
            vectors = (vectors / norms).cpu().numpy()
            for row, vector in zip(pairs[start:end], vectors):
                values[row[0], row[1]] = vector
            # Commit data before progress: an interruption only repeats a microbatch.
            values.flush()
            manifest['completed_persons'] = end
            write_json(manifest_path, manifest)
            if start == 0 or (start // args.batch_size) % 100 == 0 or end == len(pairs):
                print('Encoded persons: {}/{}'.format(end, len(pairs)), flush=True)
    global_features(values, valid)  # validate before publishing completion
    del values
    manifest['files'] = {name: file_identity(root / name) for name in (
        'person_features.npy', 'person_valid.npy')}
    manifest['complete'] = True
    write_json(manifest_path, manifest)
    print('Complete: {} samples, {} persons, {} dimensions; {}'.format(
        count, len(pairs), dim, root), flush=True)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--captions', nargs='+', required=True)
    parser.add_argument('--clip-model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    return parser


if __name__ == '__main__':
    run(get_parser().parse_args())
