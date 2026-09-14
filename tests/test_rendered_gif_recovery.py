"""Recover only duplicates proved by frame durations; reject actual lost frames."""
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'vlm_pilot'))
from caption_qwen3vl_rendered_train_transformers import load_gif_frames, close_frames


class GifRecoveryTests(unittest.TestCase):
    def write_gif(self, root, durations):
        # Distinct palette colors prevent the fixture writer merging stored frames.
        frames = [Image.new('RGB', (12, 12), (i * 7, 255 - i * 7, i * 3))
                  for i in range(len(durations))]
        path = Path(root) / 'preview.gif'
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=durations, optimize=False, disposal=2)
        close_frames(frames)
        return path

    def test_observed_31_frame_3850ms_case(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_gif(root, [120] * 12 + [250] + [120] * 18)
            frames = load_gif_frames(path, 32, 8.)
            self.assertEqual(len(frames), 32)
            self.assertEqual(frames[12].tobytes(), frames[13].tobytes())
            self.assertNotEqual(frames[13].tobytes(), frames[14].tobytes())
            self.assertIsNot(frames[12], frames[13])
            close_frames(frames)

    def test_two_merged_pairs_and_triple(self):
        for durations in ([250, 250] + [120] * 28, [370] + [120] * 29):
            with self.subTest(durations=durations[:2]), tempfile.TemporaryDirectory() as root:
                frames = load_gif_frames(self.write_gif(root, durations), 32, 8.)
                self.assertEqual(len(frames), 32)
                close_frames(frames)

    def test_normal_32_frames_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            frames = load_gif_frames(self.write_gif(root, [120] * 32), 32, 8.)
            self.assertEqual(len({f.tobytes() for f in frames}), 32)
            close_frames(frames)

    def test_missing_or_unexplained_timing_still_fails(self):
        for durations in ([120] * 31, [240] + [120] * 30, [0] * 31):
            with self.subTest(first=durations[0]), tempfile.TemporaryDirectory() as root:
                with self.assertRaisesRegex(ValueError, 'frame count mismatch'):
                    load_gif_frames(self.write_gif(root, durations), 32, 8.)

    def test_writer_round_trip_preserves_every_original_position(self):
        from render_qwen3vl_train import save_gif
        frames = [Image.new('RGB', (12, 12), (i * 7, 255 - i * 7, 0)) for i in range(31)]
        frames.insert(10, frames[9].copy())
        try:
            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / 'preview.gif'
                save_gif(frames, path, 8.)
                recovered = load_gif_frames(path, 32, 8.)
                try:
                    self.assertEqual([f.tobytes() for f in recovered], [f.tobytes() for f in frames])
                finally:
                    close_frames(recovered)
        finally:
            close_frames(frames)


if __name__ == '__main__':
    unittest.main()
