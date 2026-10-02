"""CPU checks for LP metric aggregation without requiring Torch or timm."""
import ast
import contextlib
from collections import defaultdict
import io
from pathlib import Path
import types
import unittest

import numpy as np


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def float(self):
        return Tensor(self.values.astype(np.float64))

    def long(self):
        return Tensor(self.values.astype(np.int64))

    def to(self, *args, **kwargs):
        return self

    def item(self):
        return self.values.item()

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def mean(self, dim):
        return Tensor(self.values.mean(axis=dim))


class Meter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += value * n
        self.count += n

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger:
    def __init__(self, **kwargs):
        self.meters = defaultdict(Meter)

    def update(self, **values):
        for name, value in values.items():
            self.meters[name].update(value)

    def log_every(self, batches, *args):
        return iter(batches)

    def synchronize_between_processes(self):
        pass

    def __getattr__(self, name):
        return self.meters[name]


class NoGrad(contextlib.ContextDecorator):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def cross_entropy_per_sample(logits, targets):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(targets)), targets]


class CrossEntropyLoss:
    def __call__(self, logits, targets):
        return Tensor(cross_entropy_per_sample(logits.values, targets.values).mean())


def accuracy(logits, targets, topk):
    order = np.argsort(-logits.values, axis=1)
    return [Tensor(100.0 * (order[:, :k] == targets.values[:, None]).any(axis=1).mean())
            for k in topk]


class Model:
    num_classes = 6

    def __init__(self):
        self.evaluating = False

    def eval(self):
        self.evaluating = True

    def __call__(self, inputs):
        if not self.evaluating:
            raise AssertionError('LP evaluation must put the model in eval mode')
        return inputs


class ClassMeter:
    def __init__(self, *args):
        pass

    def update(self, *args):
        pass

    def print_class_accuracy(self):
        pass


def load_evaluators():
    # Execute the real evaluation functions while leaving their heavy imports out.
    path = Path(__file__).resolve().parents[1] / 'engine_linprobe.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('evaluate', 'evaluate_original')]
    torch = types.SimpleNamespace(
        no_grad=NoGrad,
        nn=types.SimpleNamespace(CrossEntropyLoss=CrossEntropyLoss),
        cat=lambda tensors, dim: Tensor(np.concatenate([t.values for t in tensors], axis=dim)))
    namespace = {'torch': torch, 'misc': types.SimpleNamespace(MetricLogger=MetricLogger),
                 'accuracy': accuracy, 'class_accuracy_meter': ClassMeter}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in ('evaluate', 'evaluate_original')}


class LinprobeEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluators = load_evaluators()
        cls.logits = np.array([[5, 4, 3, 2, 1, 0], [0, 5, 4, 3, 2, 1],
                               [0, 1, 5, 4, 3, 2], [5, 4, 3, 2, 1, 0],
                               [8, 5, 4, 3, 2, -8]], dtype=np.float64)
        cls.targets = np.array([0, 1, 2, 3, 5])

    def run_evaluate(self, name, sizes, multiple_views=False):
        batches = []
        offset = 0
        for size in sizes:
            logits = self.logits[offset:offset + size]
            targets = Tensor(self.targets[offset:offset + size])
            images = ([Tensor(logits + .25), Tensor(logits - .25)]
                      if multiple_views else Tensor(logits))
            batches.append((images, targets) if name == 'evaluate_original'
                           else (images, images, targets, None))
            offset += size
        self.assertEqual(offset, len(self.targets))
        with contextlib.redirect_stdout(io.StringIO()):
            return self.evaluators[name](batches, Model(), 'cpu')

    def assert_metrics(self, stats):
        expected_loss = cross_entropy_per_sample(self.logits, self.targets).mean()
        self.assertAlmostEqual(stats['loss'], expected_loss, places=12)
        self.assertAlmostEqual(stats['acc1'], 60.0, places=12)
        self.assertAlmostEqual(stats['acc5'], 80.0, places=12)

    def test_small_last_batch_has_its_sample_weight(self):
        losses = cross_entropy_per_sample(self.logits, self.targets)
        old_batch_average = (losses[:4].mean() + losses[4:].mean()) / 2
        self.assertGreater(abs(old_batch_average - losses.mean()), 1.0)
        for name in self.evaluators:
            with self.subTest(evaluator=name):
                self.assert_metrics(self.run_evaluate(name, [4, 1]))

    def test_metrics_are_independent_of_batch_partition(self):
        for name in self.evaluators:
            with self.subTest(evaluator=name):
                self.assert_metrics(self.run_evaluate(name, [2, 3]))
                self.assert_metrics(self.run_evaluate(name, [5]))

    def test_multi_view_logits_keep_the_same_sample_weight(self):
        self.assert_metrics(self.run_evaluate('evaluate', [4, 1], multiple_views=True))


if __name__ == '__main__':
    unittest.main()
