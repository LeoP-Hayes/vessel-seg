from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from typing import get_args

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vessel_reproduction import data_manifest as dm


class TestDataManifestInterface(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name) / "data"
        for split in ("train", "val", "test"):
            split_dir = self.data_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(2):
                img = np.full((32, 32), idx * 40 + 60, dtype=np.uint8)
                cv2.imwrite(str(split_dir / f"{idx}.png"), img)
        self.sample_path = self.data_dir / "test" / "0.png"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_manifest_fields_match_dataclass(self) -> None:
        expected = tuple(field.name for field in dataclasses.fields(dm.ManifestRecord))
        self.assertEqual(dm.DEFAULT_MANIFEST_FIELDS, expected)

    def test_error_code_enum_complete(self) -> None:
        expected = {
            "PATH_NOT_FOUND",
            "UNSUPPORTED_EXT",
            "PERMISSION_DENIED",
            "DECODE_FAILED",
            "INVALID_SHAPE",
            "EMPTY_IMAGE",
            "INTERNAL_ERROR",
        }
        self.assertEqual(set(get_args(dm.ErrorCode)), expected)

    def test_read_grayscale_image(self) -> None:
        img, meta = dm.read_grayscale_image(self.sample_path)
        self.assertEqual(img.ndim, 2)
        self.assertEqual(meta.width, img.shape[1])
        self.assertEqual(meta.height, img.shape[0])

    def test_discover_samples_split_mode(self) -> None:
        samples = dm.discover_samples(self.data_dir, options=dm.DiscoverOptions(input_mode="split_dirs", recursive=False))
        self.assertGreater(len(samples), 0)
        self.assertIn(samples[0].split, ("train", "val", "test"))

    def test_build_manifest_and_process_batch(self) -> None:
        samples = dm.discover_samples(self.data_dir, options=dm.DiscoverOptions(input_mode="single_dir", recursive=False))
        small = samples[:3]
        records = dm.build_manifest(small, options=dm.ManifestBuildOptions(read_image=True, include_checksum=True))
        self.assertEqual(len(records), len(small))
        self.assertTrue(all(r.read_status == "ok" for r in records))

        summary, recs = dm.process_batch(small, options=dm.ProcessBatchOptions(on_error="record", max_errors=5))
        self.assertEqual(summary.total, len(small))
        self.assertEqual(summary.ok, len(small))
        self.assertEqual(len(recs), len(small))

    def test_manifest_to_rows(self) -> None:
        samples = dm.discover_samples(self.data_dir, options=dm.DiscoverOptions(input_mode="single_dir", recursive=True))
        recs = dm.build_manifest(samples[:1], options=dm.ManifestBuildOptions(read_image=False, include_checksum=False))
        rows = dm.manifest_to_rows(recs)
        self.assertEqual(len(rows), 1)
        self.assertIn("sample_id", rows[0])


if __name__ == "__main__":
    unittest.main()
