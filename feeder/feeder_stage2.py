"""Two-view NTU feeder for the unified ReSA/OSE Stage2 protocol."""

import math
import random

import numpy as np
from torch.utils.data import Dataset

from feeder import tools


def shear(data_numpy, amplitude=0.5):
    first = [random.uniform(-amplitude, amplitude) for _ in range(3)]
    second = [random.uniform(-amplitude, amplitude) for _ in range(3)]
    matrix = np.array([
        [1.0, first[0], second[0]],
        [first[1], 1.0, second[1]],
        [first[2], second[2], 1.0],
    ]).T
    transformed = np.dot(data_numpy.transpose(1, 2, 3, 0), matrix)
    return transformed.transpose(3, 0, 1, 2)


def temporal_crop(data_numpy, padding_ratio=6):
    _, frames, _, _ = data_numpy.shape
    if padding_ratio <= 0:
        return data_numpy
    padding = frames // int(padding_ratio)
    if padding <= 0:
        return data_numpy
    start = np.random.randint(0, padding * 2 + 1)
    padded = np.concatenate([
        data_numpy[:, :padding][:, ::-1],
        data_numpy,
        data_numpy[:, -padding:][:, ::-1],
    ], axis=1)
    return padded[:, start:start + frames]


def random_rotate(data_numpy):
    def rotation_matrix(axis, angle):
        sine, cosine = math.sin(angle), math.cos(angle)
        if axis == 0:
            matrix = np.array([
                [1.0, 0.0, 0.0],
                [0.0, cosine, sine],
                [0.0, -sine, cosine],
            ])
        elif axis == 1:
            matrix = np.array([
                [cosine, 0.0, -sine],
                [0.0, 1.0, 0.0],
                [sine, 0.0, cosine],
            ])
        else:
            matrix = np.array([
                [cosine, sine, 0.0],
                [-sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ])
        return matrix.T

    sequence = data_numpy.transpose(1, 2, 3, 0).copy()
    primary_axis = random.randint(0, 2)
    for axis in range(3):
        maximum = 30.0 if axis == primary_axis else 1.0
        angle = math.radians(random.uniform(0.0, maximum))
        sequence = np.matmul(sequence, rotation_matrix(axis, angle))
    return sequence.transpose(3, 0, 1, 2)


class FeederStage2(Dataset):
    """Load MAMP NPZ data and return two independently augmented views."""

    _SUPPORTED_AUGMENTATIONS = ('temporal_crop', 'shear', 'rotation')

    def __init__(
        self,
        data_path,
        split='train',
        window_size=120,
        base_p_interval=(0.95,),
        shear_amplitude=0.5,
        temporal_padding_ratio=6,
        augmentation_methods=('temporal_crop', 'shear', 'rotation'),
        augmentation_probability=0.5,
        use_mmap=True,
    ):
        if split != 'train':
            raise ValueError('FeederStage2 only supports the training split')
        if window_size <= 0:
            raise ValueError('window_size must be positive')
        self.data_path = data_path
        self.window_size = int(window_size)
        self.base_p_interval = tuple(float(value)
                                     for value in base_p_interval)
        self.shear_amplitude = float(shear_amplitude)
        self.temporal_padding_ratio = int(temporal_padding_ratio)
        self.augmentation_methods = tuple(augmentation_methods)
        unknown = [
            name for name in self.augmentation_methods
            if name not in self._SUPPORTED_AUGMENTATIONS]
        if unknown:
            raise ValueError(
                'Unsupported Stage2 augmentations: {}'.format(unknown))
        if len(set(self.augmentation_methods)) != len(
                self.augmentation_methods):
            raise ValueError('Stage2 augmentations must not repeat')
        self.augmentation_probability = float(augmentation_probability)
        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError('augmentation_probability must be in [0, 1]')

        archive = np.load(data_path, mmap_mode='r' if use_mmap else None)
        if 'x_train' not in archive.files or 'y_train' not in archive.files:
            raise KeyError('Stage2 NPZ requires x_train and y_train')
        data = archive['x_train']
        targets = archive['y_train']
        if data.ndim != 3:
            raise ValueError('x_train must have shape [N,T,150]')
        if targets.ndim != 2 or targets.shape[0] != data.shape[0]:
            raise ValueError('y_train must be a matching one-hot array')
        positives = targets > 0
        if not np.all(positives.sum(axis=1) == 1):
            raise ValueError('Every Stage2 label must be one-hot')
        self.label = np.where(positives)[1].astype(np.int64, copy=False)
        samples, frames, coordinates = data.shape
        if coordinates != 2 * 25 * 3:
            raise ValueError('Stage2 currently expects NTU 25-joint data')
        self.data = data.reshape(samples, frames, 2, 25, 3).transpose(
            0, 4, 1, 3, 2)
        archive.close()

    def __len__(self):
        return len(self.label)

    def get_base_sample(self, index):
        sample = np.asarray(self.data[int(index)]).copy()
        valid_frames = int(np.sum(
            sample.sum(axis=0).sum(axis=-1).sum(axis=-1) != 0))
        if valid_frames <= 0:
            raise ValueError('Stage2 sample {} has no valid frame'.format(
                int(index)))
        return tools.valid_crop_resize(
            sample, valid_frames, self.base_p_interval, self.window_size)

    def _apply_augmentation(self, sample, name):
        if name == 'temporal_crop':
            return temporal_crop(sample, self.temporal_padding_ratio)
        if name == 'shear':
            return shear(sample, self.shear_amplitude)
        if name == 'rotation':
            return random_rotate(sample)
        raise ValueError('Unsupported Stage2 augmentation {}'.format(name))

    def augment(self, sample):
        sample = np.asarray(sample).copy()
        for name in self.augmentation_methods:
            if random.random() < self.augmentation_probability:
                sample = self._apply_augmentation(sample, name)
        return np.asarray(sample, dtype=np.float32)

    def get_augmented_sample(self, index):
        return self.augment(self.get_base_sample(index))

    def __getitem__(self, index):
        base = self.get_base_sample(index)
        return (
            self.augment(base),
            self.augment(base),
            int(self.label[index]),
            int(index),
        )
