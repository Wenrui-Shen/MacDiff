"""Portable, label-free CLIP cache validation (NumPy only at training time)."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

PROTOCOL = 'macdiff_clip_person_cache_v1'


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
    for name in ('person_features.npy', 'person_valid.npy'):
        if manifest['files'][name] != file_identity(root / name):
            raise ValueError('Cache file checksum mismatch: ' + name)
    values = np.load(root / 'person_features.npy', mmap_mode='r', allow_pickle=False)
    valid = np.load(root / 'person_valid.npy', allow_pickle=False)
    if values.shape != (count, 2, dim) or values.dtype != np.float32:
        raise ValueError('Unexpected cached feature shape/dtype')
    return global_features(values, valid), manifest
