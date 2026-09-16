"""Portable, label-free CLIP cache validation (NumPy only at training time)."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

PROTOCOL = 'macdiff_clip_token_cache_v2'
CACHE_FILES = ('person_features.npy', 'person_valid.npy', 'token_features.npy',
               'token_mask.npy', 'token_ids.npy')


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def train_shape(path):
    # Read only the NPY header; do not decompress the full skeleton array.
    with zipfile.ZipFile(path) as archive, archive.open('x_train.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Unsupported x_train NPY header version')
    if len(shape) != 3 or shape[0] < 1 or shape[2] != 150 or dtype.hasobject:
        raise ValueError('Expected NTU x_train with shape [N,T,150]')
    return tuple(shape)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def global_features(person_features, valid):
    values = np.asarray(person_features, dtype=np.float32)
    valid = np.asarray(valid)
    if (values.ndim != 3 or values.shape[1] != 2 or valid.shape != values.shape[:2]
            or valid.dtype != np.bool_ or not valid.any(axis=1).all()
            or not np.isfinite(values).all()):
        raise ValueError('Invalid person features or validity mask')
    if np.any(values[~valid] != 0):
        raise ValueError('Absent-person cache slots must be zero')
    norms = np.linalg.norm(values[valid], axis=-1)
    if not np.allclose(norms, 1., atol=1e-4):
        raise ValueError('Expected unit-normalized per-person CLIP features')
    pooled = (values * valid[..., None]).sum(axis=1) / valid.sum(axis=1)[:, None]
    norms = np.linalg.norm(pooled, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError('Degenerate global text feature')
    return np.asarray(pooled / norms, dtype=np.float32)


def load_cache(directory, data_path=None, expected_count=None):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('protocol') != PROTOCOL or manifest.get('complete') is not True:
        raise ValueError('CLIP cache is incomplete or uses a different protocol')
    count, dim = manifest['sample_count'], manifest['feature_dim']
    if expected_count is not None and count != expected_count:
        raise ValueError('Cache sample count differs from training dataset')
    if data_path is not None and manifest['identity']['data'] != file_identity(data_path):
        raise ValueError('Cache was generated for a different dataset file')
    for name in CACHE_FILES:
        if manifest['files'][name] != file_identity(root / name):
            raise ValueError('Cache file checksum mismatch: ' + name)
    values = np.load(root / 'person_features.npy', mmap_mode='r', allow_pickle=False)
    valid = np.load(root / 'person_valid.npy', allow_pickle=False)
    if values.shape != (count, 2, dim) or values.dtype != np.float32:
        raise ValueError('Unexpected cached feature shape/dtype')
    validate_token_cache(root, manifest, valid)
    return global_features(values, valid), manifest


def validate_token_rows(features, mask, ids, eos_token_id):
    """Validate complete person rows, including original positions and padding."""
    if mask.dtype != np.bool_ or ids.dtype != np.int32:
        raise ValueError('Invalid token mask/ID dtype')
    lengths = mask.sum(axis=-1)
    expected = (np.arange(mask.shape[-1])[None, :] > 0) & (
        np.arange(mask.shape[-1])[None, :] <= lengths[:, None])
    if (np.any(lengths < 1) or not np.array_equal(mask, expected)
            or np.any(ids[~mask] != -1) or np.any(ids[mask] < 0)
            or np.any(ids[np.arange(len(ids)), lengths] != eos_token_id)):
        raise ValueError('Invalid token positions, padding, or EOS alignment')
    if (not np.isfinite(features).all() or np.any(features[~mask] != 0)
            or not np.allclose(np.linalg.norm(features[mask].astype(np.float32), axis=-1),
                               1., atol=2e-3)):
        raise ValueError('Invalid cached token feature values')


def validate_token_cache(root, manifest, person_valid):
    count, length, dim = manifest['sample_count'], manifest['context_length'], manifest['feature_dim']
    features = np.load(root / 'token_features.npy', mmap_mode='r', allow_pickle=False)
    mask = np.load(root / 'token_mask.npy', mmap_mode='r', allow_pickle=False)
    ids = np.load(root / 'token_ids.npy', mmap_mode='r', allow_pickle=False)
    if (features.shape != (count, 2, length, dim) or features.dtype != np.float16
            or mask.shape != (count, 2, length) or ids.shape != mask.shape
            or mask.dtype != np.bool_ or ids.dtype != np.int32):
        raise ValueError('Unexpected token cache shape/dtype')
    for start in range(0, count, 64):
        end = min(start + 64, count)
        f, m, i, valid = features[start:end], mask[start:end], ids[start:end], person_valid[start:end]
        if (not np.array_equal(m.any(axis=-1), valid) or np.any(f[~valid] != 0)
                or np.any(i[~valid] != -1)):
            raise ValueError('Token cache does not match person validity')
        validate_token_rows(f[valid], m[valid], i[valid], manifest['eos_token_id'])


class TokenFeatureCache:
    """Read-only mmap storage; pack valid tokens only for the requested batch."""
    def __init__(self, directory, global_text, manifest):
        self.global_text = global_text
        self.manifest = manifest
        self.features = np.load(Path(directory) / 'token_features.npy', mmap_mode='r', allow_pickle=False)
        self.mask = np.load(Path(directory) / 'token_mask.npy', mmap_mode='r', allow_pickle=False)

    def get_batch(self, indices):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or len(indices) == 0 or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= len(self.global_text))):
            raise ValueError('Invalid raw training indices for text lookup')
        valid = self.mask[indices]
        length = int(valid.sum(axis=(1, 2)).max())
        if length == 0:
            raise ValueError('A text batch contains no valid tokens')
        dim, context = self.features.shape[-1], self.features.shape[2]
        tokens = np.zeros((len(indices), length, dim), dtype=np.float32)
        token_mask = np.zeros((len(indices), length), dtype=bool)
        persons = np.zeros((len(indices), length), dtype=np.int64)
        positions = np.zeros_like(persons)
        person_grid = np.broadcast_to(np.arange(2)[:, None], (2, context))
        position_grid = np.broadcast_to(np.arange(context)[None, :], (2, context))
        for row, index in enumerate(indices):
            m = valid[row]
            size = int(m.sum())
            if size == 0:
                raise ValueError('A sample contains no valid text tokens')
            tokens[row, :size] = self.features[index][m]
            token_mask[row, :size] = True
            persons[row, :size] = person_grid[m]
            positions[row, :size] = position_grid[m]
        return {'text_features': self.global_text[indices], 'text_tokens': tokens,
                'text_token_mask': token_mask, 'text_person_ids': persons,
                'text_positions': positions}


def load_token_cache(directory, data_path=None, expected_count=None):
    global_text, manifest = load_cache(directory, data_path, expected_count)
    return TokenFeatureCache(directory, global_text, manifest)
