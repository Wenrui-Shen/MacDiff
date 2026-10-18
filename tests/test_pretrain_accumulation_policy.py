"""Execute the engine's update statements with scalar gradients, without Torch.

These checks cover accumulation boundaries and auxiliary state updates, not
tensor autograd, AMP numerics, or DDP behavior.
"""
import ast
from pathlib import Path
import time
import types
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'engine_pretrain.py'


def update_block(function_name):
    function = next(node for node in ast.parse(SOURCE.read_text(encoding='utf-8')).body
                    if isinstance(node, ast.FunctionDef) and node.name == function_name)
    loop = next(node for node in function.body if isinstance(node, ast.For))
    finite_check = next(index for index, node in enumerate(loop.body)
                        if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                        and isinstance(node.test.operand, ast.Call)
                        and isinstance(node.test.operand.func, ast.Attribute)
                        and node.test.operand.func.attr == 'isfinite')
    boundary_check = next(index for index in range(finite_check + 1, len(loop.body))
                          if isinstance(loop.body[index], ast.If)
                          and isinstance(loop.body[index].test, ast.Name)
                          and loop.body[index].test.id == 'accumulation_boundary')
    module = ast.Module(body=loop.body[finite_check + 1:boundary_check + 1], type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(SOURCE), 'exec')


class Optimizer:
    def __init__(self):
        self.grad = 0.
        self.applied_gradients = []

    def zero_grad(self):
        self.grad = 0.


class Scaler:
    def __init__(self, skip_attempts=()):
        self._scaler = self
        self.scale = 8.
        self.attempts = 0
        self.skip_attempts = skip_attempts

    def get_scale(self):
        return self.scale

    def __call__(self, loss, optimizer, parameters=None, update_grad=True):
        # A linear scalar loss has a gradient equal to its coefficient.
        optimizer.grad += loss
        if update_grad:
            self.attempts += 1
            if self.attempts in self.skip_attempts:
                self.scale /= 2
            else:
                optimizer.applied_gradients.append(optimizer.grad)


class Model:
    def __init__(self):
        self.text_updates = 0
        self.teacher_updates = 0
        self.enqueued = []

    def parameters(self):
        return []

    def update_text_target(self):
        self.text_updates += 1

    def update_momentum_encoder(self):
        self.teacher_updates += 1

    def enqueue_ose(self, features, ids):
        self.enqueued.append(tuple(item.value for item in ids))


class Bank:
    def __init__(self):
        self.updated_ids = []

    def update_from_model(self, indices, model, device, batches=None):
        self.updated_ids.append(tuple(indices))


class DetachedValue:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self


def run_epoch(ose=False, accum=2, count=5, text=False, bank=None,
              optimizer=None, model=None, scaler=None):
    optimizer = optimizer or Optimizer()
    model = model or Model()
    scaler = scaler or Scaler()
    optimizer.zero_grad()  # The actual engine starts each epoch this way.
    env = dict(accum_iter=accum, steps_this_epoch=count, optimizer=optimizer,
               model=model, model_without_ddp=model, loss_scaler=scaler,
               text_features=object() if text else None, text_target_bank=bank,
               pending_target_indices=[], pending_target_batches=[],
               pending_teacher_features=[], pending_source_ids=[],
               target_update_seconds=0., device=types.SimpleNamespace(type='cpu'),
               time=time, concat_all_gather=lambda value: value,
               torch=types.SimpleNamespace(cat=lambda values, dim=0: tuple(values)))
    block = update_block('train_one_epoch_ose' if ose else 'train_one_epoch_macdiff')
    for step in range(count):
        env.update(data_iter_step=step, loss=float(step + 1),
                   teacher_features=DetachedValue(step), source_ids=DetachedValue(step))
        if text and bank is not None:
            env['pending_target_indices'].append(step)
            env['pending_target_batches'].append(step)
        exec(block, env)
    return optimizer, model, scaler


class UpstreamAccumulationTests(unittest.TestCase):
    def test_partial_window_is_not_applied_or_carried_into_next_epoch(self):
        optimizer, model, scaler = run_epoch()
        self.assertEqual(optimizer.applied_gradients, [1.5, 3.5])
        self.assertEqual(optimizer.grad, 2.5)
        run_epoch(count=2, optimizer=optimizer, model=model, scaler=scaler)
        self.assertEqual(optimizer.applied_gradients, [1.5, 3.5, 1.5])

    def test_text_bank_updates_only_complete_windows(self):
        bank = Bank()
        optimizer, _, _ = run_epoch(text=True, bank=bank)
        self.assertEqual(optimizer.applied_gradients, [1.5, 3.5])
        self.assertEqual(bank.updated_ids, [(0, 1), (2, 3)])

    def test_skipped_amp_step_does_not_update_text_target(self):
        optimizer, model, _ = run_epoch(text=True, scaler=Scaler(skip_attempts=(1,)))
        self.assertEqual(optimizer.applied_gradients, [3.5])
        self.assertEqual(model.text_updates, 1)

    def test_ose_teacher_and_queue_follow_successful_complete_windows(self):
        for skipped, expected_ids in (((), [(0, 1), (2, 3)]), ((1,), [(2, 3)])):
            with self.subTest(skipped=skipped):
                optimizer, model, _ = run_epoch(ose=True, scaler=Scaler(skipped))
                self.assertEqual(model.enqueued, expected_ids)
                self.assertEqual(model.teacher_updates, len(expected_ids))
                self.assertEqual(len(optimizer.applied_gradients), len(expected_ids))

    def test_accum_one_updates_every_batch_in_both_paths(self):
        for ose in (False, True):
            with self.subTest(ose=ose):
                optimizer, _, _ = run_epoch(ose=ose, accum=1)
                self.assertEqual(optimizer.applied_gradients, [1., 2., 3., 4., 5.])


if __name__ == '__main__':
    unittest.main()
