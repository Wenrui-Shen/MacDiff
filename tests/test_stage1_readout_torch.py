"""Torch extraction checks; run on the server alongside NumPy regressions."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for encoder/extraction checks')
class ExtractionTests(unittest.TestCase):
    def test_frozen_extraction_cache_keeps_weights_and_bn(self):
        import torch
        from stage1_readout import extract_cache

        class Dataset(torch.utils.data.Dataset):
            label = [0] * 6 + [1] * 6
            sample_name = [str(i) for i in range(12)]

            def __len__(self):
                return 12

            def __getitem__(self, index):
                return torch.tensor([float(index), 1., -1.]), self.label[index]

        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            checkpoint = parent / 'checkpoint.pth'
            torch.save(model.state_dict(), checkpoint)
            args = argparse.Namespace(
                checkpoint=str(checkpoint), cache_dir=str(parent / 'cache'), device='cpu',
                batch_size=5, workers=0, exemplar_seeds=[0, 1], exemplar_caches=[],
                seed=17, max_train=0, max_eval=0)
            with contextlib.redirect_stdout(io.StringIO()):
                extract_cache(args, 'synthetic', model, (Dataset(), Dataset()),
                              lambda m, x: {'I': m(x)}, {})
            manifest = json.loads((parent / 'cache/manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['branches'], {'I': 4})
            self.assertEqual(np.load(parent / 'cache/train_I.npy').shape, (12, 4))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(before[name], value), name)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))


    def test_macdiff_keeps_all_tokens_and_both_people(self):
        import torch
        from model.transformer_downstream import Transformer
        from compare_stage1_readouts import forward_features, load_encoder

        args = dict(dim_in=3, dim_feat=8, depth=1, num_heads=2, num_frames=8,
                    num_joints=25, patch_size=1, t_patch_size=4, protocol='linprobe2')
        source = Transformer(**args).eval()
        model = Transformer(**args).eval()
        model.head.fc = torch.nn.Identity()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stage1.pth'
            torch.save({'model': source.state_dict()}, path)
            load_encoder(model, str(path))
        captured = {}
        hook = model.norm.register_forward_hook(
            lambda module, inputs, output: captured.update(tokens=output.detach().clone()))
        x = torch.randn(2, 3, 8, 25, 2)
        with torch.no_grad():
            actual = forward_features(model, x)['J']
        hook.remove()
        self.assertEqual(tuple(captured['tokens'].shape), (4, 50, 8))
        expected = captured['tokens'].reshape(2, 2, 2, 25, 8).mean(dim=(1, 2)).reshape(2, 200)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
