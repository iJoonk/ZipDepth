import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from zipdepth.data.dataset import LargeScaleDepthDataset
from zipdepth.data.transforms import PairedShortestSideCrop


def _save_bytes(path: Path, values: list[str]) -> None:
    width = max(map(len, values)) + 1
    np.save(path, np.asarray(values, dtype=f'S{width}'))


class KittiBaselineTest(unittest.TestCase):
    def test_paired_transform_is_square_float32_and_bilinear(self):
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        depth = np.asarray(
            [[0.0, 10.0, 0.0, 10.0], [0.0, 10.0, 0.0, 10.0]],
            dtype=np.float32,
        )
        transform = PairedShortestSideCrop(size=4, flip_probability=0.0)
        np.random.seed(0)
        output_image, output_depth = transform(image, depth)
        self.assertEqual(output_image.shape, (4, 4, 3))
        self.assertEqual(output_depth.shape, (4, 4))
        self.assertEqual(output_depth.dtype, np.float32)
        self.assertTrue(np.any((output_depth > 0.0) & (output_depth < 10.0)))

    def test_phase10b_pt_depth_bypasses_uint16_quantization(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            image_path = tmp_path / 'rgb.png'
            cache_path = tmp_path / 'teacher.pt'
            image = np.zeros((3, 5, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)
            depth = torch.linspace(0.125, 620.75, 15, dtype=torch.float32).reshape(1, 1, 3, 5)
            sample_id = 'date/drive/frame'
            image_sha = 'a' * 64
            teacher_identity = 'b' * 64
            cache_version = 'phase10b_teacher_depth_v1'
            torch.save({
                'metadata': {
                    'sample_id': sample_id,
                    'image_sha256': image_sha,
                    'teacher_identity': teacher_identity,
                    'cache_version': cache_version,
                    'shape': [1, 1, 3, 5],
                    'dtype': 'float32',
                    'finite': True,
                },
                'depth': depth,
            }, cache_path)

            prefix = tmp_path / 'index'
            _save_bytes(Path(f'{prefix}_rgb.npy'), [str(image_path)])
            _save_bytes(Path(f'{prefix}_depth.npy'), [str(cache_path)])
            _save_bytes(Path(f'{prefix}_domain.npy'), ['kitti'])
            _save_bytes(Path(f'{prefix}_sample_id.npy'), [sample_id])
            _save_bytes(Path(f'{prefix}_image_sha256.npy'), [image_sha])
            metadata = {
                'total_samples': 1,
                'strict_loading': True,
                'teacher_identity': teacher_identity,
                'cache_version': cache_version,
            }
            Path(f'{prefix}_metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            Path(f'{prefix}.json').write_text('{}', encoding='utf-8')

            sample = LargeScaleDepthDataset(str(prefix) + '.json')[0]
            torch.testing.assert_close(sample['depth'], depth.squeeze(0))
            self.assertEqual(sample['depth'].dtype, torch.float32)
            self.assertEqual(sample['depth_scale'].item(), 1.0)
            self.assertEqual(sample['depth'].max().item(), 620.75)


if __name__ == '__main__':
    unittest.main()
