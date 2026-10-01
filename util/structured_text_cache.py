"""Sentence-level global/body-region cache validation, without Torch at training time."""
import json
from pathlib import Path
import zipfile

import numpy as np

from .clip_text_cache import file_identity, train_shape


PROTOCOL = 'macdiff_clip_sentence_cache_v3'
LOCAL_PARTS = ('head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg')
CACHE_FILES = ('person_features.npy', 'person_valid.npy', 'token_features.npy', 'token_mask.npy')
DEFINITION = {'representation': 'projected_sentence_text_embeds', 'normalization': 'unit_rms',
              'local_parts': list(LOCAL_PARTS), 'local_position_ids': list(range(1, 7)),
              'person_alignment': 'raw_npz_person_slots', 'dtype': 'float32'}


def read_captions(paths, sample_count):
    """Read text-only JSON arrays or JSONL plus their separate provenance files."""
    rows, versions, metadata_files = {}, set(), []
    for path in map(Path, paths):
        metadata_path = path.with_suffix('.metadata.json')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if (metadata.get('schema_version') != 'macdiff.caption_text.v2'
                or metadata.get('caption_schema') != 'global_local'):
            raise ValueError('Expected global/local caption metadata: ' + str(metadata_path))
        version = (metadata.get('model'), metadata.get('prompt', {}).get('sha256'),
                   metadata.get('requested_revision'))
        if not version[0] or not version[1]:
            raise ValueError('Missing caption model/prompt identity')
        versions.add(version)
        metadata_files.append(metadata_path)
        with path.open(encoding='utf-8') as stream:
            records = json.load(stream) if path.suffix.lower() == '.json' else (
                json.loads(line) for line in stream if line.strip())
            if path.suffix.lower() == '.json' and not isinstance(records, list):
                raise ValueError('Expected a batch JSON array: ' + str(path))
            for record in records:
                if not isinstance(record, dict) or set(record) != {'sample_index', 'persons'}:
                    raise ValueError('Expected a text-only caption sample')
                index = record['sample_index']
                if type(index) is not int or not 0 <= index < sample_count:
                    raise ValueError('Caption index is outside x_train: ' + repr(index))
                if index in rows:
                    raise ValueError('Duplicate caption sample_index: ' + str(index))
                persons = record['persons']
                if not isinstance(persons, list) or len(persons) not in (1, 2):
                    raise ValueError('Invalid caption persons: ' + str(index))
                normalized = []
                for ordinal, person in enumerate(persons):
                    if (not isinstance(person, dict)
                            or set(person) != {'person_index', 'global', 'local'}
                            or type(person['person_index']) is not int
                            or person['person_index'] != ordinal
                            or not isinstance(person['local'], dict)
                            or set(person['local']) != set(LOCAL_PARTS)):
                        raise ValueError('Invalid person/body-region alignment: ' + str(index))
                    sentences = [person['global']] + [person['local'][part] for part in LOCAL_PARTS]
                    if any(not isinstance(text, str) or not text.strip() for text in sentences):
                        raise ValueError('Empty global/local sentence: ' + str(index))
                    normalized.append({'person_index': ordinal, 'global': sentences[0].strip(),
                                       'local': {part: sentences[i + 1].strip()
                                                 for i, part in enumerate(LOCAL_PARTS)}})
                rows[index] = {'sample_index': index, 'persons': normalized}
    if len(versions) != 1:
        raise ValueError('Use exactly one caption model, prompt and revision')
    missing = sorted(set(range(sample_count)) - set(rows))
    if missing:
        raise ValueError('Missing captions: {} (first 20: {})'.format(len(missing), missing[:20]))
    return [rows[i] for i in range(sample_count)], list(next(iter(versions))), metadata_files


def train_person_valid(path, batch_size=64):
    """Stream only x_train from NPZ to check actual raw person slots, without labels."""
    shape = train_shape(path)
    result = np.zeros((shape[0], 2), dtype=bool)
    with zipfile.ZipFile(path) as archive, archive.open('x_train.npy') as stream:
        version = np.lib.format.read_magic(stream)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        stored_shape, fortran, dtype = read_header(stream)
        if fortran or dtype.kind not in 'fiu':
            raise ValueError('Expected a C-order numeric x_train array')
        for start in range(0, shape[0], batch_size):
            count = min(batch_size, shape[0] - start)
            size = count * shape[1] * 150 * dtype.itemsize
            block = stream.read(size)
            if len(block) != size:
                raise ValueError('Truncated x_train array')
            values = np.frombuffer(block, dtype=dtype).reshape(count, shape[1], 2, 25, 3)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite raw skeleton values')
            result[start:start + count] = np.any(values != 0, axis=(1, 3, 4))
    if not result.any(axis=1).all():
        raise ValueError('x_train contains a sample without any skeleton person')
    return result


def unit_rms(values):
    values = np.asarray(values, dtype=np.float32)
    energy = np.mean(values * values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(energy).all() or np.any(energy < 1e-12):
        raise ValueError('Invalid CLIP sentence vectors')
    return values / np.sqrt(energy)


def validate_rows(global_values, local_values):
    for values in (global_values, local_values):
        if (not np.isfinite(values).all()
                or not np.allclose(np.mean(values * values, axis=-1), 1., atol=2e-5)):
            raise ValueError('Expected finite unit-RMS sentence vectors')


def validate_training_definition(manifest, model_args, feeder=None):
    if manifest.get('protocol') != PROTOCOL:
        return
    if (model_args.get('text_target_norm', 'none') != 'rms'
            or model_args.get('text_target_mode', 'remap') == 'remap'):
        raise ValueError('Sentence cache already uses unit RMS; select an RMS fixed_clip/ema_remap/sample_target_blend configuration')
    if feeder is not None and getattr(feeder, 'flip', False):
        raise ValueError('Body-region sentence cache requires flip=False until crop/flip metadata and text remapping are implemented')


def validate_cache(directory, data_path=None, expected_count=None, skip_full_validation=False,
                   manifest=None):
    root = Path(directory)
    if manifest is None:
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('protocol') != PROTOCOL or manifest.get('complete') is not True:
        raise ValueError('Sentence cache is incomplete or uses a different protocol')
    if (manifest.get('definition') != DEFINITION
            or manifest.get('identity', {}).get('definition') != DEFINITION
            or manifest.get('context_length') != 7):
        raise ValueError('Sentence cache has an incompatible region order/representation')
    count, dim = manifest['sample_count'], manifest['feature_dim']
    if type(count) is not int or type(dim) is not int or min(count, dim) < 1:
        raise ValueError('Invalid sentence cache sample count/dimension')
    if expected_count is not None and count != expected_count:
        raise ValueError('Cache sample count differs from training dataset')
    if data_path is not None:
        identity = manifest['identity']['data']
        if skip_full_validation:
            if Path(data_path).stat().st_size != identity['bytes']:
                raise ValueError('Dataset file size differs from cached dataset')
        elif file_identity(data_path) != identity:
            raise ValueError('Cache was generated for a different dataset file')
    arrays = {}
    specifications = {'person_features.npy': ((count, 2, dim), np.float32),
                      'person_valid.npy': ((count, 2), np.bool_),
                      'token_features.npy': ((count, 2, 6, dim), np.float32),
                      'token_mask.npy': ((count, 2, 6), np.bool_)}
    try:
        for name, (shape, dtype) in specifications.items():
            path = root / name
            if path.stat().st_size != manifest['files'][name]['bytes']:
                raise ValueError('Cache file size mismatch: ' + name)
            if not skip_full_validation and file_identity(path) != manifest['files'][name]:
                raise ValueError('Cache file checksum mismatch: ' + name)
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
            if arrays[name].shape != shape or arrays[name].dtype != dtype:
                raise ValueError('Unexpected sentence cache shape/dtype: ' + name)
        if not skip_full_validation:
            for start in range(0, count, 128):
                valid = arrays['person_valid.npy'][start:start + 128]
                mask = arrays['token_mask.npy'][start:start + 128]
                global_values = arrays['person_features.npy'][start:start + 128]
                local_values = arrays['token_features.npy'][start:start + 128]
                if (not valid.any(axis=1).all()
                        or not np.array_equal(mask, np.repeat(valid[..., None], 6, axis=-1))
                        or np.any(global_values[~valid] != 0) or np.any(local_values[~valid] != 0)):
                    raise ValueError('Sentence cache has invalid person slots/masks/padding')
                validate_rows(global_values[valid], local_values[valid])
    finally:
        for values in arrays.values():
            values._mmap.close()
    return manifest
