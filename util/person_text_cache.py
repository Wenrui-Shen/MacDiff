"""Person-aligned training reader for unchanged v2 CLIP cache files."""
import json
from pathlib import Path
import numpy as np
from .clip_text_cache import CACHE_FILES, PROTOCOL, load_cache


class PersonTokenFeatureCache:
    def __init__(self, directory, manifest):
        root = Path(directory)
        self.manifest = manifest
        self.global_text = np.load(root / 'person_features.npy', mmap_mode='r', allow_pickle=False)
        self.features = np.load(root / 'token_features.npy', mmap_mode='r', allow_pickle=False)
        self.mask = np.load(root / 'token_mask.npy', mmap_mode='r', allow_pickle=False)

    def get_batch(self, indices, one_person=True):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= len(self.global_text))):
            raise ValueError('Invalid raw training indices for text lookup')
        people = 1 if one_person else 2
        valid = self.mask[indices, :people]
        length = max(1, int(valid.sum(axis=-1).max()))
        dim = self.features.shape[-1]
        rows = len(indices) * people
        tokens = np.zeros((rows, length, dim), dtype=np.float32)
        mask = np.zeros((rows, length), dtype=bool)
        persons = np.zeros((rows, length), dtype=np.int64)
        positions = np.zeros_like(persons)
        for sample, index in enumerate(indices):
            for person in range(people):
                row = sample * people + person
                pos = np.flatnonzero(valid[sample, person])
                count = len(pos)
                tokens[row, :count] = self.features[index, person, pos]
                mask[row, :count] = True
                persons[row, :count] = person
                positions[row, :count] = pos
        return dict(text_features=np.asarray(self.global_text[indices, :people], dtype=np.float32).reshape(rows, dim),
                    text_tokens=tokens, text_token_mask=mask,
                    text_person_ids=persons, text_positions=positions)


def load_person_token_cache(directory, data_path=None, expected_count=None,
                            skip_full_validation=False):
    # Keep extraction helpers unchanged so existing v2 provenance remains valid.
    if skip_full_validation:
        manifest = _validate_cache_headers(directory, data_path, expected_count)
    else:
        _, manifest = load_cache(directory, data_path, expected_count)
    return PersonTokenFeatureCache(directory, manifest)


def _validate_cache_headers(directory, data_path, expected_count):
    """Fast reuse of a trusted cache; no content hashes or feature-row scans.

    File sizes and NPY headers cannot detect same-size content changes. The
    returned identity is the saved manifest, not newly verified content hashes.
    """
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('protocol') != PROTOCOL or manifest.get('complete') is not True:
        raise ValueError('CLIP cache is incomplete or uses a different protocol')
    count, dim, length = (manifest[key] for key in ('sample_count', 'feature_dim', 'context_length'))
    if expected_count is not None and count != expected_count:
        raise ValueError('Cache sample count differs from training dataset')
    if data_path is not None and Path(data_path).stat().st_size != manifest['identity']['data']['bytes']:
        raise ValueError('Dataset file size differs from the cached dataset identity')
    arrays = {
        'person_features.npy': ((count, 2, dim), np.float32),
        'person_valid.npy': ((count, 2), np.bool_),
        'token_features.npy': ((count, 2, length, dim), np.float16),
        'token_mask.npy': ((count, 2, length), np.bool_),
        'token_ids.npy': ((count, 2, length), np.int32),
    }
    for name in CACHE_FILES:
        path = root / name
        if path.stat().st_size != manifest['files'][name]['bytes']:
            raise ValueError('Cache file size mismatch: ' + name)
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            shape, dtype = arrays[name]
            if values.shape != shape or values.dtype != dtype:
                raise ValueError('Unexpected cache shape/dtype: ' + name)
        finally:
            values._mmap.close()
    return manifest
