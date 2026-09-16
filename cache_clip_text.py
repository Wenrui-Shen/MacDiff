"""Cache all accepted NTU training captions with a frozen local HF CLIP encoder."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_stage1_text_geometry import read_captions
from util.clip_text_cache import (PROTOCOL, CACHE_FILES, file_identity, global_features,
                                  load_cache, train_shape, write_json,
                                  validate_token_cache, validate_token_rows)


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


def encode_batch(model, tokenizer, texts, device, context_length):
    batch = tokenizer(texts, padding='max_length', max_length=context_length,
                      truncation=False, return_tensors='pt').to(device)
    output = model(**batch)
    # Use the same frozen projection as the sentence embedding, now at every
    # final-layer token position. This does not assert token-level CLIP alignment.
    global_values = output.text_embeds.float().cpu().numpy()
    tokens = model.text_projection(output.last_hidden_state).float().cpu().numpy()
    mask = batch['attention_mask'].cpu().numpy().astype(bool)
    ids = batch['input_ids'].cpu().numpy().astype(np.int32)
    mask[:, 0] = False  # exclude BOS; retain content tokens and the real EOS
    norms = np.linalg.norm(global_values, axis=-1, keepdims=True)
    token_norms = np.linalg.norm(tokens, axis=-1, keepdims=True)
    if (not np.isfinite(global_values).all() or not np.isfinite(tokens).all()
            or np.any(norms < 1e-8) or np.any(token_norms[..., 0][mask] < 1e-8)):
        raise ValueError('Invalid CLIP features')
    global_values = global_values / norms
    tokens = np.where(mask[..., None], tokens / np.maximum(token_norms, 1e-8), 0)
    ids = np.where(mask, ids, -1).astype(np.int32)
    return global_values, tokens.astype(np.float16), mask, ids


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
    tokenized = tokenizer(texts, padding=False, truncation=False)['input_ids']
    lengths = [len(ids) for ids in tokenized]
    too_long = [(pairs[i][:2], n) for i, n in enumerate(lengths)
                if n > config.max_position_embeddings]
    if too_long:
        raise ValueError('Captions exceed CLIP token limit (no truncation): ' + str(too_long[:20]))
    if any(len(ids) < 2 or ids[0] != tokenizer.bos_token_id
           or ids[-1] != tokenizer.eos_token_id for ids in tokenized):
        raise ValueError('Expected CLIP BOS/content/EOS tokenization')
    dim, count = config.projection_dim, len(samples)
    context = config.max_position_embeddings
    valid = np.zeros((count, 2), dtype=bool)
    for i, person, _ in pairs:
        valid[i, person] = True
    if manifest is None:
        root.mkdir(parents=True, exist_ok=True)
        values = np.lib.format.open_memmap(root / 'person_features.npy', mode='w+',
                                          dtype=np.float32, shape=(count, 2, dim))
        values[:] = 0
        values.flush()
        # Newly created mmap files are zero initialized, including absent people.
        tokens = np.lib.format.open_memmap(root / 'token_features.npy', mode='w+',
            dtype=np.float16, shape=(count, 2, context, dim))
        token_mask = np.lib.format.open_memmap(root / 'token_mask.npy', mode='w+',
            dtype=bool, shape=(count, 2, context))
        token_ids = np.lib.format.open_memmap(root / 'token_ids.npy', mode='w+',
            dtype=np.int32, shape=(count, 2, context))
        token_ids[:] = -1
        for array in (tokens, token_mask, token_ids):
            array.flush()
        np.save(root / 'person_valid.npy', valid, allow_pickle=False)
        write_json(root / 'samples.json', samples)
        manifest = {'protocol': PROTOCOL, 'identity': identity, 'complete': False,
                    'sample_count': count, 'feature_dim': dim, 'completed_persons': 0,
                    'context_length': context, 'eos_token_id': tokenizer.eos_token_id,
                    'token_representation': 'last_hidden_state @ frozen text_projection; per-token L2; FP16; BOS/padding zero',
                    'total_persons': len(pairs), 'caption_counts': counts,
                    'representation': 'projected text_embeds; FP32; per-person L2',
                    'duplicate_policy': 'last accepted; invalid/error records ignored'}
        write_json(manifest_path, manifest)
    else:
        values = np.load(root / 'person_features.npy', mmap_mode='r+', allow_pickle=False)
        tokens = np.load(root / 'token_features.npy', mmap_mode='r+', allow_pickle=False)
        token_mask = np.load(root / 'token_mask.npy', mmap_mode='r+', allow_pickle=False)
        token_ids = np.load(root / 'token_ids.npy', mmap_mode='r+', allow_pickle=False)
        if (values.shape != (count, 2, dim) or values.dtype != np.float32
                or tokens.shape != (count, 2, context, dim) or tokens.dtype != np.float16
                or token_mask.shape != (count, 2, context) or token_mask.dtype != np.bool_
                or token_ids.shape != token_mask.shape or token_ids.dtype != np.int32
                or manifest['context_length'] != context
                or manifest['eos_token_id'] != tokenizer.eos_token_id
                or not np.array_equal(np.load(root / 'person_valid.npy'), valid)
                or manifest['total_persons'] != len(pairs)
                or not 0 <= manifest['completed_persons'] <= len(pairs)):
            raise ValueError('Incomplete cache shape/progress does not match captions')
        done = pairs[:manifest['completed_persons']]
        for start in range(0, len(done), 64):
            rows = done[start:start + 64]
            indices, persons = [p[0] for p in rows], [p[1] for p in rows]
            saved = values[indices, persons]
            if not np.isfinite(saved).all() or not np.allclose(np.linalg.norm(saved, axis=1), 1, atol=1e-4):
                raise ValueError('Completed cache rows are corrupt')
            validate_token_rows(tokens[indices, persons], token_mask[indices, persons],
                                token_ids[indices, persons], tokenizer.eos_token_id)
    with torch.inference_mode():
        for start in range(manifest['completed_persons'], len(pairs), args.batch_size):
            end = min(start + args.batch_size, len(pairs))
            vectors, token_vectors, masks, ids = encode_batch(
                model, tokenizer, texts[start:end], args.device, context)
            validate_token_rows(token_vectors, masks, ids, tokenizer.eos_token_id)
            for j, (row, vector) in enumerate(zip(pairs[start:end], vectors)):
                values[row[0], row[1]] = vector
                tokens[row[0], row[1]] = token_vectors[j]
                token_mask[row[0], row[1]] = masks[j]
                token_ids[row[0], row[1]] = ids[j]
            # Commit data before progress: an interruption only repeats a microbatch.
            for array in (values, tokens, token_mask, token_ids):
                array.flush()
            manifest['completed_persons'] = end
            write_json(manifest_path, manifest)
            if start == 0 or (start // args.batch_size) % 100 == 0 or end == len(pairs):
                print('Encoded persons: {}/{}'.format(end, len(pairs)), flush=True)
    global_features(values, valid)  # validate before publishing completion
    validate_token_cache(root, manifest, valid)
    del values, tokens, token_mask, token_ids
    manifest['files'] = {name: file_identity(root / name) for name in CACHE_FILES}
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
