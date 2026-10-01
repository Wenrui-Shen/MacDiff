"""One host-shared RAM target bank; SQLite is only the checkpoint format.

Linux uses an mmap on /dev/shm. Once all local ranks attach, its files are
unlinked so the OS releases the RAM even if the training processes are killed.
Windows uses a named anonymous mapping. A short process lock makes a whole
accumulation-window update atomic, including repeated raw sample indices.
"""
import atexit
from contextlib import contextmanager
import json
import mmap
import os
from pathlib import Path
import socket
import sqlite3
import tempfile
import time
import uuid

import numpy as np

from .sample_text_target_bank import SampleTextTargetBank, PROTOCOL, _cache_identity


def _token_counts(cache):
    if hasattr(cache, 'mask'):
        return np.asarray(cache.mask[:, 0].sum(axis=-1), dtype=np.int64)
    # Small synthetic caches and callers without an exposed mmap mask.
    counts = []
    for start in range(0, len(cache.global_text), 256):
        indices = np.arange(start, min(start + 256, len(cache.global_text)))
        counts.extend(cache.get_batch(indices, one_person=True)['text_token_mask'].sum(axis=1))
    return np.asarray(counts, dtype=np.int64)


class SharedMemoryTextTargetBank(SampleTextTargetBank):
    """Same sample histories and v1 snapshots as the SQLite backend."""

    @classmethod
    def prepare(cls, path, cache, ratio, resume=None, rank=0, barrier=None):
        path = Path(path)
        bank = None
        storage = None
        try:
            if rank == 0:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not resume and path.exists():
                    raise FileExistsError('Refusing to reuse target bank: ' + str(path))
                if resume and not cls.checkpoint_path(resume).is_file():
                    raise FileNotFoundError('Missing target bank for resume: '
                                            + str(cls.checkpoint_path(resume)))
                counts = _token_counts(cache)
                offset = ((len(counts) + 63) // 64) * 64
                vectors = int(np.where(counts > 0, counts + 1, 0).sum())
                size = max(64, offset + vectors * int(cache.features.shape[-1]) * 4)
                storage = cls._create_storage(size, path.parent)
                descriptor = {'hostname': socket.gethostname(), 'bytes': size,
                              'vector_offset': offset, 'vector_count': vectors,
                              'storage': storage, 'metadata': cls._metadata(cache, ratio)}
                temporary = path.with_name(path.name + '.tmp')
                with open(temporary, 'w', encoding='utf-8') as file:
                    json.dump(descriptor, file)
                os.replace(str(temporary), str(path))
                bank = cls(path, cache, ratio)
                if resume:
                    bank._restore(cls.checkpoint_path(resume))
            if barrier is not None:
                barrier()
            if rank != 0:
                bank = cls(path, cache, ratio)
            if barrier is not None:
                barrier()
            if rank == 0 and os.name != 'nt':
                cls._unlink_storage(storage)
            atexit.register(bank.close)
            return bank
        except Exception:
            if bank is not None:
                bank.close()
            if rank == 0 and storage is not None:
                cls._unlink_storage(storage)
            raise

    @staticmethod
    def _metadata(cache, ratio):
        if not 0 < float(ratio) <= 1:
            raise ValueError('Target update ratio must be in (0, 1]')
        return {'protocol': PROTOCOL, 'cache_identity': _cache_identity(cache),
                'sample_count': str(len(cache.global_text)),
                'dimension': str(cache.features.shape[-1]),
                'update_ratio': repr(float(ratio))}

    @staticmethod
    def _create_storage(size, directory):
        if os.name == 'nt':
            root = Path(tempfile.mkdtemp(prefix='macdiff-target-', dir=str(directory)))
            storage = {'kind': 'windows', 'name': 'macdiff-target-' + uuid.uuid4().hex,
                       'root': str(root), 'lock': str(root / 'lock')}
        else:
            root_dir = Path('/dev/shm')
            if not root_dir.is_dir():
                raise RuntimeError('Shared target bank requires /dev/shm; '
                                   'use --text_target_bank_backend sqlite on this host')
            available = os.statvfs(str(root_dir))
            if available.f_bavail * available.f_frsize < size:
                raise RuntimeError('Insufficient /dev/shm space for {:.2f} GiB target bank; '
                                   'increase tmpfs capacity or use --text_target_bank_backend sqlite'
                                   .format(size / 1024 ** 3))
            root = Path(tempfile.mkdtemp(prefix='macdiff-target-', dir=str(root_dir)))
            storage = {'kind': 'tmpfs', 'root': str(root),
                       'vectors': str(root / 'vectors'), 'lock': str(root / 'lock')}
        try:
            if storage['kind'] == 'tmpfs':
                with open(storage['vectors'], 'w+b') as file:
                    file.truncate(size)
                    if hasattr(os, 'posix_fallocate'):
                        os.posix_fallocate(file.fileno(), 0, size)
            with open(storage['lock'], 'wb') as file:
                file.write(b'\0')
            return storage
        except Exception:
            SharedMemoryTextTargetBank._unlink_storage(storage)
            raise

    @staticmethod
    def _unlink_storage(storage):
        # Only explicit files in this bank's own newly created directory.
        for name in ('vectors', 'lock'):
            if name in storage:
                try:
                    Path(storage[name]).unlink()
                except (FileNotFoundError, PermissionError):
                    pass
        try:
            Path(storage['root']).rmdir()
        except OSError:
            pass

    def __init__(self, path, cache, ratio):
        self.path, self.cache, self.ratio = Path(path), cache, float(ratio)
        self.dim = int(cache.features.shape[-1])
        with open(self.path, encoding='utf-8') as file:
            descriptor = json.load(file)
        if descriptor['metadata'] != self._metadata(cache, ratio):
            raise ValueError('Target bank metadata differs from this run/cache')
        if descriptor['hostname'] != socket.gethostname():
            raise ValueError('Shared target bank only supports DDP ranks on the same host')
        self.counts = _token_counts(cache)
        self.offsets = np.concatenate(([0], np.cumsum(
            np.where(self.counts > 0, self.counts + 1, 0))))
        offset = ((len(self.counts) + 63) // 64) * 64
        expected_bytes = max(64, offset + int(self.offsets[-1]) * self.dim * 4)
        if (descriptor['vector_count'] != self.offsets[-1]
                or descriptor['vector_offset'] != offset
                or descriptor['bytes'] != expected_bytes):
            raise ValueError('Target bank memory layout differs from this cache')
        self.storage = descriptor['storage']
        self._file = None
        self._mapping = None
        self._lock_file = None
        self._initialized = self._vectors = None
        try:
            self._lock_file = open(self.storage['lock'], 'r+b')
            if self.storage['kind'] == 'windows':
                self._mapping = mmap.mmap(-1, expected_bytes, tagname=self.storage['name'],
                                          access=mmap.ACCESS_WRITE)
            else:
                self._file = open(self.storage['vectors'], 'r+b')
                self._mapping = mmap.mmap(self._file.fileno(), expected_bytes,
                                          access=mmap.ACCESS_WRITE)
            self._initialized = np.ndarray((len(self.counts),), dtype=np.uint8,
                                           buffer=self._mapping)
            self._vectors = np.ndarray((int(self.offsets[-1]), self.dim), dtype=np.float32,
                                      buffer=self._mapping, offset=offset)
        except Exception:
            self.close()
            raise

    @contextmanager
    def _locked(self):
        if os.name == 'nt':
            import msvcrt
            deadline = time.monotonic() + 60
            while True:
                self._lock_file.seek(0)
                try:
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Timed out locking shared text target bank')
                    time.sleep(.001)
        else:
            import fcntl
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)

    def close(self):
        self._initialized = self._vectors = None
        for name in ('_mapping', '_file', '_lock_file'):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.close()
                setattr(self, name, None)
        if getattr(self, 'storage', {}).get('kind') == 'windows':
            self._unlink_storage(self.storage)

    def initialized_samples(self):
        with self._locked():
            return int(np.count_nonzero(self._initialized))

    def _batch_indices(self, indices, arrays):
        indices = np.asarray(indices, dtype=np.int64)
        if (indices.ndim != 1 or len(indices) != len(arrays['text_features'])
                or np.any(indices < 0) or np.any(indices >= len(self.counts))):
            raise ValueError('Target indices and cached text are not aligned')
        counts = arrays['text_token_mask'].sum(axis=1)
        if np.any((counts != 0) & (counts != self.counts[indices])):
            raise ValueError('Target local token count differs from this cache')
        return indices, counts

    def get_batch(self, indices, arrays):
        indices, counts = self._batch_indices(indices, arrays)
        global_target = np.zeros_like(arrays['text_features'], dtype=np.float32)
        local_target = np.zeros_like(arrays['text_tokens'], dtype=np.float32)
        with self._locked():
            for row, index in enumerate(indices):
                count = int(counts[row])
                if not count:
                    continue
                values = (self._vectors[self.offsets[index]:self.offsets[index + 1]]
                          if self._initialized[index] else self._fixed_row(arrays, row))
                global_target[row] = values[0]
                local_target[row, :count] = values[1:]
        return {'text_target_global': global_target, 'text_target_tokens': local_target}

    def apply_updates(self, indices, arrays, online_global, online_local):
        indices, counts = self._batch_indices(indices, arrays)
        online_global = np.asarray(online_global, dtype=np.float32)
        online_local = np.asarray(online_local, dtype=np.float32)
        if (online_global.shape != arrays['text_features'].shape
                or online_local.shape != arrays['text_tokens'].shape
                or not np.isfinite(online_global).all() or not np.isfinite(online_local).all()):
            raise ValueError('Invalid online text targets')
        with self._locked():
            for row, index in enumerate(indices):
                count = int(counts[row])
                if not count:
                    continue
                slot = self._vectors[self.offsets[index]:self.offsets[index + 1]]
                old = slot if self._initialized[index] else self._fixed_row(arrays, row)
                current = np.concatenate((online_global[row:row + 1],
                                          online_local[row, :count]), axis=0)
                slot[:] = (1. - self.ratio) * old + self.ratio * current
                self._initialized[index] = 1

    def _restore(self, source):
        connection = sqlite3.connect(str(source), timeout=60)
        try:
            metadata = dict(connection.execute('SELECT name, value FROM metadata'))
            if metadata != self._metadata(self.cache, self.ratio):
                raise ValueError('Target bank metadata differs from this run/cache')
            for index, blob in connection.execute('SELECT sample_id, vectors FROM targets'):
                if index < 0 or index >= len(self.counts) or self.counts[index] == 0:
                    raise ValueError('Corrupt target-bank sample index')
                self._vectors[self.offsets[index]:self.offsets[index + 1]] = self._decode(
                    blob, int(self.counts[index]))
                self._initialized[index] = 1
        finally:
            connection.close()

    def snapshot(self, checkpoint):
        destination = self.checkpoint_path(checkpoint)
        temporary = destination.with_name(destination.name + '.tmp')
        if destination.exists() or temporary.exists():
            raise FileExistsError('Target-bank snapshot exists: ' + str(destination))
        connection = sqlite3.connect(str(temporary), timeout=60)
        try:
            # Only the unpublished temporary file uses these settings. Flush the
            # completed snapshot before its atomic rename; no per-row fsync.
            connection.execute('PRAGMA journal_mode=OFF')
            connection.execute('PRAGMA synchronous=OFF')
            connection.execute('CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
            connection.execute('CREATE TABLE targets (sample_id INTEGER PRIMARY KEY, vectors BLOB NOT NULL)')
            connection.executemany('INSERT INTO metadata VALUES (?, ?)',
                                   sorted(self._metadata(self.cache, self.ratio).items()))
            with self._locked():
                connection.executemany('INSERT INTO targets VALUES (?, ?)', (
                    (int(index), self._vectors[self.offsets[index]:self.offsets[index + 1]].tobytes())
                    for index in np.flatnonzero(self._initialized)))
                connection.commit()
        finally:
            connection.close()
        with open(temporary, 'r+b') as file:
            os.fsync(file.fileno())
        os.replace(str(temporary), str(destination))
