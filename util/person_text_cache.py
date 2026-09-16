"""Person-aligned training reader for unchanged v2 CLIP cache files."""
from pathlib import Path
import numpy as np
from .clip_text_cache import load_cache


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


def load_person_token_cache(directory, data_path=None, expected_count=None):
    # Keep extraction helpers unchanged so existing v2 provenance remains valid.
    _, manifest = load_cache(directory, data_path, expected_count)
    return PersonTokenFeatureCache(directory, manifest)
