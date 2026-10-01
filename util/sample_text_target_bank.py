"""Persistent per-sample text denoising targets for Stage1.

Each row starts at fixed RMS CLIP. After a successful optimizer step, only
samples in that step move toward the *current output* of the online remap:
target[i] = (1 - ratio) * target[i] + ratio * remap(clip[i]).
The bank is shared by local DDP processes through SQLite transactions and is
snapshotted next to each model checkpoint. No EMA copy of the network exists.
"""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import shutil
import sqlite3
import uuid

import numpy as np


PROTOCOL = 'sample_text_target_bank_v1'


def _cache_identity(cache):
    identity = {key: cache.manifest[key]
                for key in ('protocol', 'identity', 'files')}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _unit_rms(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.sqrt(np.maximum(np.mean(values * values, axis=-1,
                                                keepdims=True), 1e-12))


@contextmanager
def _snapshot_file(checkpoint, destination):
    """Publish a fresh bank without replacing a bank paired with a model.

    A crash can leave a complete bank or legacy .tmp before the model is saved.
    Keep that bank until its replacement is fully written, then publish atomically.
    """
    checkpoint, destination = Path(checkpoint), Path(destination)
    if checkpoint.exists():
        raise FileExistsError('Refusing to replace target bank for existing checkpoint: '
                              + str(checkpoint))
    temporary = destination.with_name(destination.name + '.tmp-' + uuid.uuid4().hex)
    with open(temporary, 'xb'):
        pass
    try:
        yield temporary
        with open(temporary, 'r+b') as file:
            os.fsync(file.fileno())
        # A model may have appeared while the snapshot was being constructed.
        if checkpoint.exists():
            raise FileExistsError('Refusing to replace target bank for existing checkpoint: '
                                  + str(checkpoint))
        os.replace(str(temporary), str(destination))
        # Remove only the known legacy temporary file, after the new bank exists.
        legacy_temporary = destination.with_name(destination.name + '.tmp')
        try:
            legacy_temporary.unlink()
        except (FileNotFoundError, PermissionError):
            pass
    finally:
        try:
            temporary.unlink()
        except (FileNotFoundError, PermissionError):
            pass


class SampleTextTargetBank:
    """One atomic variable-length target row per raw training sample index."""

    @staticmethod
    def checkpoint_path(checkpoint):
        checkpoint = Path(checkpoint)
        return checkpoint.with_name(checkpoint.stem + '-target-bank.sqlite')

    @classmethod
    def prepare(cls, path, cache, ratio, resume=None, rank=0, barrier=None):
        """Rank zero creates/restores the DB before all ranks open it."""
        path = Path(path)
        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            if resume:
                source = cls.checkpoint_path(resume)
                if not source.is_file():
                    raise FileNotFoundError('Missing target bank for resume: ' + str(source))
                shutil.copyfile(str(source), str(path))
            else:
                if path.exists():
                    raise FileExistsError('Refusing to reuse target bank: ' + str(path))
                connection = sqlite3.connect(str(path), timeout=60)
                try:
                    connection.execute('CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
                    connection.execute('CREATE TABLE targets (sample_id INTEGER PRIMARY KEY, vectors BLOB NOT NULL)')
                    metadata = {
                        'protocol': PROTOCOL,
                        'cache_identity': _cache_identity(cache),
                        'sample_count': str(len(cache.global_text)),
                        'dimension': str(cache.features.shape[-1]),
                        'update_ratio': repr(float(ratio)),
                    }
                    connection.executemany('INSERT INTO metadata VALUES (?, ?)',
                                           sorted(metadata.items()))
                    connection.commit()
                finally:
                    connection.close()
        if barrier is not None:
            barrier()
        return cls(path, cache, ratio)

    def __init__(self, path, cache, ratio):
        self.path = Path(path)
        self.cache = cache
        self.ratio = float(ratio)
        if not 0 < self.ratio <= 1:
            raise ValueError('Target update ratio must be in (0, 1]')
        self.dim = int(cache.features.shape[-1])
        self.connection = sqlite3.connect(str(path), timeout=60)
        self.connection.execute('PRAGMA busy_timeout=60000')
        self.connection.execute('PRAGMA synchronous=FULL')
        metadata = dict(self.connection.execute('SELECT name, value FROM metadata'))
        expected = {
            'protocol': PROTOCOL,
            'cache_identity': _cache_identity(cache),
            'sample_count': str(len(cache.global_text)),
            'dimension': str(self.dim),
            'update_ratio': repr(self.ratio),
        }
        if metadata != expected:
            self.connection.close()
            raise ValueError('Target bank metadata differs from this run/cache')

    def close(self):
        self.connection.close()

    def initialized_samples(self):
        return int(self.connection.execute('SELECT COUNT(*) FROM targets').fetchone()[0])

    def _stored_rows(self, indices):
        indices = [int(index) for index in indices]
        placeholders = ','.join('?' for _ in indices)
        if not indices:
            return {}
        return dict(self.connection.execute(
            'SELECT sample_id, vectors FROM targets WHERE sample_id IN ('
            + placeholders + ')', indices))

    def _fixed_row(self, arrays, row):
        count = int(arrays['text_token_mask'][row].sum())
        if count < 1:
            raise ValueError('Text target needs at least one local token')
        combined = np.concatenate((
            arrays['text_features'][row:row + 1],
            arrays['text_tokens'][row, :count].astype(np.float32)), axis=0)
        return _unit_rms(combined)

    def _decode(self, blob, count):
        expected = (count + 1) * self.dim
        values = np.frombuffer(blob, dtype=np.float32)
        if values.size != expected or not np.isfinite(values).all():
            raise ValueError('Corrupt target-bank row')
        return values.reshape(count + 1, self.dim).copy()

    def get_batch(self, indices, arrays):
        """Return the target that existed before this optimizer update."""
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) != len(arrays['text_features']):
            raise ValueError('Target indices and cached text are not aligned')
        stored = self._stored_rows(indices)
        global_target = np.zeros_like(arrays['text_features'], dtype=np.float32)
        local_target = np.zeros_like(arrays['text_tokens'], dtype=np.float32)
        for row, index in enumerate(indices):
            count = int(arrays['text_token_mask'][row].sum())
            # Inactive cropped people are excluded from both losses and bank
            # updates. Their cache row may legitimately contain no text tokens.
            if count == 0:
                continue
            values = (self._decode(stored[int(index)], count) if int(index) in stored
                      else self._fixed_row(arrays, row))
            global_target[row] = values[0]
            local_target[row, :count] = values[1:]
        return {'text_target_global': global_target,
                'text_target_tokens': local_target}

    def apply_updates(self, indices, arrays, online_global, online_local):
        """Blend output vectors atomically, preserving distinct sample history."""
        indices = np.asarray(indices, dtype=np.int64)
        online_global = np.asarray(online_global, dtype=np.float32)
        online_local = np.asarray(online_local, dtype=np.float32)
        if (len(indices) != len(arrays['text_features'])
                or online_global.shape != arrays['text_features'].shape
                or online_local.shape != arrays['text_tokens'].shape
                or not np.isfinite(online_global).all()
                or not np.isfinite(online_local).all()):
            raise ValueError('Invalid online text targets')
        try:
            self.connection.execute('BEGIN IMMEDIATE')
            stored = self._stored_rows(indices)
            updates = []
            for row, index in enumerate(indices):
                count = int(arrays['text_token_mask'][row].sum())
                old = (self._decode(stored[int(index)], count)
                       if int(index) in stored else self._fixed_row(arrays, row))
                current = np.concatenate((online_global[row:row + 1],
                                          online_local[row, :count]), axis=0)
                blended = ((1. - self.ratio) * old + self.ratio * current).astype(np.float32)
                encoded = blended.tobytes(order='C')
                updates.append((int(index), encoded))
                stored[int(index)] = encoded
            self.connection.executemany(
                'INSERT OR REPLACE INTO targets (sample_id, vectors) VALUES (?, ?)', updates)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def update_from_model(self, indices, model, device, batches=None):
        """Use post-step weights, reusing cached inputs from the accumulation window."""
        import torch

        if len(indices) == 0:
            return
        if batches is None:
            indices = np.asarray(indices, dtype=np.int64)
            arrays = self.cache.get_batch(indices, one_person=True)
            tensors = {key: torch.from_numpy(arrays[key]).to(device)
                       for key in ('text_features', 'text_tokens', 'text_token_mask')}
            batches = [(indices, arrays, tensors, np.ones(len(indices), dtype=bool))]
        pieces = []
        with torch.no_grad():
            for batch_indices, arrays, tensors, active in batches:
                if not active.any():
                    continue
                features = tensors['text_features']
                tokens = tensors['text_tokens']
                valid = tensors['text_token_mask']
                global_online = model.text_remap(model.fixed_clip_target(features))
                local_online = model.text_remap(model.fixed_clip_target(tokens, valid))
                local_online = local_online.masked_fill(~valid[..., None], 0)
                pieces.append((
                    np.asarray(batch_indices, dtype=np.int64)[active],
                    {key: arrays[key][active] for key in
                     ('text_features', 'text_tokens', 'text_token_mask')},
                    global_online.cpu().numpy()[active],
                    local_online.cpu().numpy()[active]))
        if not pieces:
            return
        # Different micro-batches can have different local padding lengths.
        indices = np.concatenate([piece[0] for piece in pieces])
        length = max(piece[1]['text_tokens'].shape[1] for piece in pieces)
        arrays = {
            'text_features': np.concatenate([piece[1]['text_features'] for piece in pieces]),
            'text_tokens': np.zeros((len(indices), length, self.dim), dtype=np.float32),
            'text_token_mask': np.zeros((len(indices), length), dtype=bool),
        }
        online_global = np.concatenate([piece[2] for piece in pieces])
        online_local = np.zeros_like(arrays['text_tokens'])
        start = 0
        for batch_indices, values, _, local in pieces:
            stop = start + len(batch_indices)
            width = local.shape[1]
            arrays['text_tokens'][start:stop, :width] = values['text_tokens']
            arrays['text_token_mask'][start:stop, :width] = values['text_token_mask']
            online_local[start:stop, :width] = local
            start = stop
        self.apply_updates(indices, arrays, online_global, online_local)

    def snapshot(self, checkpoint):
        """Atomic SQLite backup matched to one model checkpoint epoch."""
        destination = self.checkpoint_path(checkpoint)
        with _snapshot_file(checkpoint, destination) as temporary:
            backup = sqlite3.connect(str(temporary), timeout=60)
            try:
                self.connection.backup(backup)
            finally:
                backup.close()
