"""경로·분할·표본 구성 회귀 검사. 실제 학습 데이터나 PyTorch는 사용하지 않는다."""

import copy
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from detector_data.preflight import create_smoke_package, inspect_package, image_location
from detector_data.settings import DEFAULT_SETTINGS, REPOSITORY_ROOT, load_settings, resolve_data_paths


def write_png(path):
    def chunk(kind, value):
        return struct.pack("!I", len(value)) + kind + value + struct.pack("!I", zlib.crc32(kind + value))
    header = struct.pack("!2I5B", 32, 32, 8, 2, 0, 0, 0)
    rows = b"".join(b"\0" + bytes([220, 220, 220]) * 32 for _ in range(32))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "datasets"
        (self.package / "annotations").mkdir(parents=True)
        self.settings = copy.deepcopy(DEFAULT_SETTINGS)
        self.settings["package_dir"] = str(self.package)
        self.settings["image_roots"] = {"select": str(self.root / "datasets/images/select_aihub"), "codeit": str(self.root / "datasets/images/select_codeit")}
        self.rows = []
        self.payloads = {}
        for split, offset in (("train", 0), ("validation", 2)):
            images, annotations = [], []
            for j, dataset in enumerate(("select", "codeit"), start=1):
                iid = offset + j
                group = f"K-{iid * 2 - 1:06d}-{iid * 2:06d}"
                file_name = f"{dataset}/images/{group}_0_2_0_2_70_000_200.png"
                images.append({"id": iid, "width": 32, "height": 32, "file_name": file_name})
                write_png(image_location({"file_name": file_name}, {key: Path(value) for key, value in self.settings["image_roots"].items()}))
                for k in range(2):
                    annotations.append({"id": iid * 10 + k, "image_id": iid, "bbox": [k * 12, 2, 10, 20], "area": 200, "category_id": iid * 2 - 1 + k, "iscrowd": 0})
                self.rows.append({"image_id": iid, "dataset": dataset, "image_path": file_name,
                                  "file_name": Path(file_name).name, "original_image_id": iid,
                                  "annotation_count": 2, "group_id": group, "split": split})
            self.payloads[split] = {"images": images, "annotations": annotations, "categories": []}
        self.save()

    def save(self):
        for split, payload in self.payloads.items():
            (self.package / "annotations" / f"{split}.json").write_text(json.dumps(payload))
        with (self.package / "split_manifest.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_valid_package(self):
        report = inspect_package(self.settings)
        self.assertEqual(report["counts"]["train"], {"images": 2, "boxes": 4, "groups": 2})
        self.assertEqual(report["group_overlap_count"], 0)

    def test_defaults_ignore_working_directory(self):
        old = Path.cwd()
        try:
            os.chdir(self.root)
            with patch.dict(os.environ, {}, clear=True):
                package, _ = resolve_data_paths(load_settings())
            self.assertEqual(package, REPOSITORY_ROOT / "datasets")
        finally:
            os.chdir(old)

    def test_explicit_settings_override(self):
        custom = self.root / "custom.json"
        custom.write_text(json.dumps(self.settings))
        with patch.dict(os.environ, {"DETECTOR_DATA_SETTINGS": str(custom)}):
            self.assertEqual(load_settings(), self.settings)

    def test_missing_image_rejected(self):
        image_location({"file_name": self.rows[0]["image_path"]}, {key: Path(value) for key, value in self.settings["image_roots"].items()}).unlink()
        with self.assertRaises(FileNotFoundError):
            inspect_package(self.settings)

    def test_duplicate_id_rejected(self):
        self.payloads["train"]["annotations"].append(self.payloads["train"]["annotations"][0])
        self.save()
        with self.assertRaisesRegex(ValueError, "ID"):
            inspect_package(self.settings)

    def test_bounds_rejected(self):
        self.payloads["train"]["annotations"][0]["bbox"] = [0, 0, 33, 10]
        self.save()
        with self.assertRaisesRegex(ValueError, "경계"):
            inspect_package(self.settings)

    def test_area_mismatch_rejected(self):
        self.payloads["train"]["annotations"][0]["area"] = 199
        self.save()
        with self.assertRaisesRegex(ValueError, "면적"):
            inspect_package(self.settings)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            image_location({"file_name": "select/../private.png"}, {"select": self.root})

    def test_group_leak_rejected(self):
        old_group = self.rows[0]["group_id"]
        image = self.payloads["validation"]["images"][0]
        image["file_name"] = "select/images/" + old_group + "_0_2_0_2_90_000_200.png"
        self.rows[2]["group_id"] = old_group
        self.rows[2]["image_path"] = image["file_name"]
        write_png(image_location(image, {key: Path(value) for key, value in self.settings["image_roots"].items()}))
        self.save()
        with self.assertRaisesRegex(ValueError, "그룹 중복"):
            inspect_package(self.settings)

    def test_smoke_keeps_sources_all_boxes_and_originals(self):
        original = {str(p): p.read_bytes() for p in self.package.rglob("*") if p.is_file()}
        output = self.root / "subset"
        sources = create_smoke_package(self.settings, output, 2)
        for split in ("train", "validation"):
            self.assertEqual({r["dataset"] for r in sources[split]}, {"select", "codeit"})
            subset = json.loads((output / "annotations" / f"{split}.json").read_text())
            self.assertEqual(len(subset["annotations"]), 4)
        self.assertTrue(all(Path(p).read_bytes() == value for p, value in original.items()))
        report = inspect_package(dict(self.settings, package_dir=str(output)))
        self.assertEqual(report["counts"]["validation"]["boxes"], 4)

    def test_smoke_reproducible(self):
        first = create_smoke_package(self.settings, self.root / "a", 1)
        second = create_smoke_package(self.settings, self.root / "b", 1)
        self.assertEqual(first, second)

    def test_command_check_success(self):
        custom = self.root / "settings.json"
        custom.write_text(json.dumps(self.settings))
        report = self.root / "check_report.json"
        result = subprocess.run(
            [sys.executable, "-m", "detector_data", "check", "--report", str(report)],
            cwd=REPOSITORY_ROOT,
            env=dict(os.environ, DETECTOR_DATA_SETTINGS=str(custom)), capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(report.read_text())
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["source_files_unchanged"])

    def test_command_records_failure(self):
        custom = self.root / "settings.json"
        custom.write_text(json.dumps(self.settings))
        image_location({"file_name": self.rows[0]["image_path"]}, {key: Path(value) for key, value in self.settings["image_roots"].items()}).unlink()
        report = self.root / "failure.json"
        result = subprocess.run(
            [sys.executable, "-m", "detector_data", "check", "--report", str(report)],
            cwd=REPOSITORY_ROOT,
            env=dict(os.environ, DETECTOR_DATA_SETTINGS=str(custom)), capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(report.read_text())["status"], "failed")

    def test_source_paths_map_to_image_only_directories(self):
        roots = {key: Path(value) for key, value in self.settings["image_roots"].items()}
        for row in self.rows:
            path = image_location({"file_name": row["image_path"]}, roots)
            self.assertEqual(path.parent, roots[row["dataset"]])
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, Path(row["image_path"]).name)

    def test_unreferenced_image_is_not_training_data(self):
        roots = {key: Path(value) for key, value in self.settings["image_roots"].items()}
        write_png(roots["select"] / "excluded_photo.png")
        report = inspect_package(self.settings)
        self.assertEqual(report["counts"]["train"]["images"], 2)
        self.assertEqual(report["counts"]["validation"]["images"], 2)
        sources = create_smoke_package(self.settings, self.root / "unused_photo_subset", 10)
        self.assertEqual(sum(len(v) for v in sources.values()), 4)
        self.assertNotIn("excluded_photo.png", json.dumps(sources))

    def test_symlink_escape_rejected(self):
        roots = {key: Path(value) for key, value in self.settings["image_roots"].items()}
        outside = self.root / "outside"
        outside.mkdir()
        try:
            (roots["select"] / "linked").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("이 환경에서는 심볼릭 링크를 만들 수 없습니다.")
        with self.assertRaises(ValueError):
            image_location({"file_name": "select/images/linked/file.png"}, roots)

    def test_old_source_prefix_does_not_require_old_folders(self):
        self.assertFalse((self.root / "select").exists())
        self.assertFalse((self.root / "codeit").exists())
        before = {p: p.read_bytes() for p in self.package.rglob("*.json")}
        manifest_before = (self.package / "split_manifest.csv").read_bytes()
        inspect_package(self.settings)
        self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))
        self.assertEqual((self.package / "split_manifest.csv").read_bytes(), manifest_before)


if __name__ == "__main__":
    unittest.main()
