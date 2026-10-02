"""Person-aligned reader for v2 word tokens and v3 body-region sentences."""
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
        # v2 stores BOS at slot 0 (masked); v3's slot 0 is the head sentence.
        self.position_offset = 1 if manifest.get('protocol') == 'macdiff_clip_sentence_cache_v3' else 0

    def get_batch(self, indices, one_person=True):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= len(self.global_text))):
            raise ValueError('Invalid raw training indices for text lookup')
        people = 1 if one_person else 2
        valid = self.mask[indices, :people]
        counts = valid.sum(axis=-1)
        length = max(1, int(counts.max()))
        dim = self.features.shape[-1]
        rows = len(indices) * people
        sentence_rows = (self.position_offset == 1 and valid.shape[-1] == 6
                         and np.all((counts == 0) | (counts == 6)))
        if sentence_rows and length == 6:
            # v3 has six sentences for every present person. Gather only the
            # retained slots, preserving sample-major/person-minor ordering.
            tokens = np.asarray(self.features[indices, :people], dtype=np.float32).reshape(rows, 6, dim)
            mask = valid.reshape(rows, 6)
            persons = np.broadcast_to((np.arange(rows, dtype=np.int64) % people)[:, None],
                                      mask.shape).copy()
            positions = np.broadcast_to(np.arange(1, 7, dtype=np.int64), mask.shape).copy()
            absent = ~mask[:, 0]
            tokens[absent] = 0
            persons[absent] = 0
            positions[absent] = 0
        else:
            tokens = np.zeros((rows, length, dim), dtype=np.float32)
            mask = np.zeros((rows, length), dtype=bool)
            persons = np.zeros((rows, length), dtype=np.int64)
            positions = np.zeros_like(persons)
            # All-empty v3 batches keep the original single zero slot. v2 and
            # partial v3 masks still need compaction at their original positions.
            if not sentence_rows:
                for sample, index in enumerate(indices):
                    for person in range(people):
                        row = sample * people + person
                        pos = np.flatnonzero(valid[sample, person])
                        count = len(pos)
                        tokens[row, :count] = self.features[index, person, pos]
                        mask[row, :count] = True
                        persons[row, :count] = person
                        positions[row, :count] = pos + self.position_offset
        return dict(text_features=np.asarray(self.global_text[indices, :people], dtype=np.float32).reshape(rows, dim),
                    text_tokens=tokens, text_token_mask=mask,
                    text_person_ids=persons, text_positions=positions)


def load_person_token_cache(directory, data_path=None, expected_count=None,
                            skip_full_validation=False):
    # Keep v2 extraction/validation unchanged so existing provenance remains valid.
    manifest = json.loads((Path(directory) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('protocol') == 'macdiff_clip_sentence_cache_v3':
        from .structured_text_cache import validate_cache
        manifest = validate_cache(directory, data_path, expected_count, skip_full_validation)
        return PersonTokenFeatureCache(directory, manifest)
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
