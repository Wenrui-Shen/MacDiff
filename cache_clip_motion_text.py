"""Cache one global and six body-region CLIP sentence vectors per skeleton person."""
import argparse
import json
from pathlib import Path

import numpy as np

from cache_clip_text import load_encoder
from util.clip_text_cache import file_identity, train_shape, write_json
from util.structured_text_cache import (PROTOCOL, CACHE_FILES, DEFINITION, LOCAL_PARTS,
                                       read_captions, train_person_valid, unit_rms,
                                       validate_cache, validate_rows)


def build_plan(data_path, caption_paths):
    count = train_shape(data_path)[0]
    samples, version, metadata_files = read_captions(caption_paths, count)
    print('Checking raw skeleton person slots (x_train only)...', flush=True)
    valid = train_person_valid(data_path)
    supplied = np.zeros_like(valid)
    for sample in samples:
        for person in sample['persons']:
            supplied[sample['sample_index'], person['person_index']] = True
    mismatch = np.argwhere(supplied != valid)
    if len(mismatch):
        raise ValueError('Caption persons differ from raw skeleton slots (sample,person; first 20): '
                         + str(mismatch[:20].tolist()))
    return samples, valid, version, metadata_files


def encode_sentences(model, tokenizer, texts, device):
    batch = tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to(device)
    # A complete sentence's projected pooled feature, not individual BPE hidden states.
    vectors = model(**batch).text_embeds.float().cpu().numpy()
    return unit_rms(vectors)


def run(args):
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    samples, valid, version, metadata_files = build_plan(args.data_path, args.captions)
    root, clip = Path(args.output_dir), Path(args.clip_model)
    weight = clip / ('model.safetensors' if (clip / 'model.safetensors').exists()
                     else 'pytorch_model.bin')
    model_files = sorted(set(clip.glob('*.json')) | set(clip.glob('*.txt')) | {weight})
    print('Checking dataset, captions and CLIP fingerprints...', flush=True)
    identity = {'data': file_identity(args.data_path),
                'captions': [file_identity(path) for path in args.captions],
                'caption_metadata': [file_identity(path) for path in metadata_files],
                'caption_version': version, 'definition': DEFINITION,
                'clip': {path.name: file_identity(path) for path in model_files},
                'implementation': {name: file_identity(Path(__file__).parent / name)
                                   for name in ('cache_clip_motion_text.py', 'util/structured_text_cache.py',
                                                'cache_clip_text.py', 'util/clip_text_cache.py')}}
    pairs = [(sample['sample_index'], person['person_index'],
              [person['global']] + [person['local'][part] for part in LOCAL_PARTS])
             for sample in samples for person in sample['persons']]
    manifest_path = root / 'manifest.json'
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('protocol') != PROTOCOL or manifest.get('identity') != identity:
            raise ValueError('Cache provenance changed; use a new output directory')
        if manifest.get('complete') is True:
            validate_cache(root, args.data_path, len(samples))
            print('Reusing complete sentence cache: ' + str(root), flush=True)
            return
        if not args.resume:
            raise ValueError('Incomplete cache exists; pass --resume to continue')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Output directory is nonempty without a cache manifest')

    import torch
    model, tokenizer, config = load_encoder(clip, args.device)
    # Validate each sentence independently against CLIP's context, before writing output.
    for start in range(0, len(pairs), 128):
        labels = [(index, person, part) for index, person, texts in pairs[start:start + 128]
                  for part in ('global',) + LOCAL_PARTS]
        texts = [text for index, person, sentences in pairs[start:start + 128] for text in sentences]
        tokenized = tokenizer(texts, padding=False, truncation=False)['input_ids']
        for label, ids in zip(labels, tokenized):
            if len(ids) > config.max_position_embeddings:
                raise ValueError('Sentence exceeds CLIP token limit (no truncation): ' + str(label))
            if len(ids) < 3 or ids[0] != tokenizer.bos_token_id or ids[-1] != tokenizer.eos_token_id:
                raise ValueError('Expected CLIP BOS/content/EOS tokenization: ' + str(label))
    count, dim = len(samples), config.projection_dim
    specifications = {'person_features.npy': ((count, 2, dim), np.float32),
                      'token_features.npy': ((count, 2, 6, dim), np.float32),
                      'token_mask.npy': ((count, 2, 6), np.bool_)}
    arrays = {}
    try:
        if manifest is None:
            root.mkdir(parents=True, exist_ok=True)
            for name, (shape, dtype) in specifications.items():
                arrays[name] = np.lib.format.open_memmap(root / name, mode='w+', dtype=dtype, shape=shape)
                arrays[name][:] = 0
                arrays[name].flush()
            np.save(root / 'person_valid.npy', valid, allow_pickle=False)
            write_json(root / 'samples.json', samples)
            manifest = {'protocol': PROTOCOL, 'identity': identity, 'definition': DEFINITION,
                        'complete': False, 'sample_count': count, 'feature_dim': dim,
                        'context_length': 7, 'clip_context_length': config.max_position_embeddings,
                        'total_persons': len(pairs), 'completed_persons': 0,
                        'labels_read': False, 'person_slots_checked_against_x_train': True}
            write_json(manifest_path, manifest)
        else:
            for name, (shape, dtype) in specifications.items():
                arrays[name] = np.load(root / name, mmap_mode='r+', allow_pickle=False)
                if arrays[name].shape != shape or arrays[name].dtype != dtype:
                    raise ValueError('Incomplete sentence cache shape/dtype mismatch: ' + name)
            if (manifest.get('definition') != DEFINITION or manifest.get('context_length') != 7
                    or manifest.get('feature_dim') != dim or manifest.get('sample_count') != count
                    or manifest.get('total_persons') != len(pairs)
                    or not np.array_equal(np.load(root / 'person_valid.npy'), valid)
                    or type(manifest.get('completed_persons')) is not int
                    or not 0 <= manifest['completed_persons'] <= len(pairs)):
                raise ValueError('Incomplete cache metadata does not match captions')
            for start in range(0, manifest['completed_persons'], 128):
                done = pairs[start:min(start + 128, manifest['completed_persons'])]
                indices, persons = [p[0] for p in done], [p[1] for p in done]
                validate_rows(arrays['person_features.npy'][indices, persons],
                              arrays['token_features.npy'][indices, persons])
                if not arrays['token_mask.npy'][indices, persons].all():
                    raise ValueError('Completed sentence cache has missing region masks')
        with torch.inference_mode():
            for start in range(manifest['completed_persons'], len(pairs), args.batch_size):
                batch = pairs[start:start + args.batch_size]
                texts = [text for index, person, sentences in batch for text in sentences]
                vectors = encode_sentences(model, tokenizer, texts, args.device).reshape(len(batch), 7, dim)
                validate_rows(vectors[:, 0], vectors[:, 1:])
                for j, (index, person, texts) in enumerate(batch):
                    arrays['person_features.npy'][index, person] = vectors[j, 0]
                    arrays['token_features.npy'][index, person] = vectors[j, 1:]
                    arrays['token_mask.npy'][index, person] = True
                for values in arrays.values():
                    values.flush()
                # Publish progress only after every sentence of each person is committed.
                manifest['completed_persons'] = start + len(batch)
                write_json(manifest_path, manifest)
                if start == 0 or start // args.batch_size % 100 == 0 or start + len(batch) == len(pairs):
                    print('Encoded persons: {}/{}'.format(start + len(batch), len(pairs)), flush=True)
    finally:
        for values in arrays.values():
            values._mmap.close()
    manifest['files'] = {name: file_identity(root / name) for name in CACHE_FILES}
    manifest['complete'] = True
    # Validate before publishing completion; a failure leaves the on-disk progress incomplete.
    validate_cache(root, args.data_path, count, manifest=manifest)
    write_json(manifest_path, manifest)
    print('Complete: {} samples; 1 global + 6 local sentence vectors per person; {}'.format(count, root), flush=True)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--captions', nargs='+', required=True)
    parser.add_argument('--clip-model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=16, help='Persons per batch; seven sentences per person')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    return parser


if __name__ == '__main__':
    run(get_parser().parse_args())
