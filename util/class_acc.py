# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

class class_accuracy_meter:
    def __init__(self, num_classes=60):
        self.num_classes = num_classes
        self.cnt = torch.zeros(num_classes, num_classes)
        print(f'Create Class Accuracy Meter of {num_classes} classes.')

    def update(self, pred, label):
        '''
        pred: (N, C)
        label: (N,)
        '''
        N, C = pred.shape
        pred = torch.argmax(pred, dim=-1)
        for i in range(N):
            self.cnt[label[i], pred[i]] += 1

    def print_class_accuracy(self):
        print('All results:', self.cnt)
        for i in range(self.num_classes):
            print(f'Class {i+1} Acc: {self.cnt[i,i] / self.cnt[i].sum()}')


