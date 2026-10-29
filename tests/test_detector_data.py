"""합성 이미지와 라벨로 데이터 연결·실패 처리·표본 무결성을 검사한다."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

from detector_data.data_records import read_records
from detector_data.settings import DEFAULT_SETTINGS, REPOSITORY_ROOT, resolve_image_path
from detector_data.smoke_test import verify_samples


def write_png(path):
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("!2I5B", 32, 32, 8, 2, 0, 0, 0)
    rows = b"".join(b"\0" + bytes([220, 220, 220]) * 32 for _ in range(32))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class DataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.package = self.root / "dataset"
        (self.package / "annotations").mkdir(parents=True)
        self.settings = copy.deepcopy(DEFAULT_SETTINGS)
        self.settings.update(package_dir=str(self.package), image_roots={
            "select": str(self.package / "images/select_aihub"),
            "codeit": str(self.package / "images/select_codeit"),
        })
        self.rows, self.labels = [], {}
        for split, offset in (("train", 0), ("validation", 2)):
            images, annotations = [], []
            for j, dataset in enumerate(("select", "codeit"), 1):
                iid = offset + j
                group = f"K-{iid * 2 - 1:06d}-{iid * 2:06d}"
                name = f"{dataset}/images/{group}_0_2_0_2_70_000_200.png"
                images.append(dict(id=iid, file_name=name, width=32, height=32))
                write_png(resolve_image_path(name, self.settings["image_roots"]))
                for k in range(2):
                    annotations.append(dict(id=iid * 10 + k, image_id=iid,
                                            category_id=iid * 2 - 1 + k,
                                            bbox=[k * 12, 2, 10, 20], area=200, iscrowd=0))
                self.rows.append(dict(image_id=iid, dataset=dataset, image_path=name,
                                      original_image_id=iid, annotation_count=2,
                                      group_id=group, split=split))
            self.labels[split] = dict(images=images, annotations=annotations, categories=[])
        self.save()

    def save(self):
        for split, payload in self.labels.items():
            (self.package / "annotations" / f"{split}.json").write_text(json.dumps(payload))
        with (self.package / "split_manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def ann(self):
        return self.labels["train"]["annotations"][0]

    def test_valid_records_keep_all_boxes(self):
        records, label_map, summary = read_records(self.settings)
        self.assertEqual(summary["num_train_instances"], 4)
        self.assertEqual(summary["num_val_images"], 2)
        self.assertEqual(label_map["num_model_classes"], 2)
        self.assertEqual(records["train"][0]["boxes"], [[0.0, 2.0, 10.0, 22.0], [12.0, 2.0, 22.0, 22.0]])

    def test_excluded_unreferenced_image_is_not_loaded(self):
        write_png(Path(self.settings["image_roots"]["select"]) / "excluded.png")
        records, _, _ = read_records(self.settings)
        self.assertEqual(sum(map(len, records.values())), 4)

    def test_duplicate_annotation_rejected(self):
        self.labels["train"]["annotations"].append(copy.deepcopy(self.ann()))
        self.save()
        with self.assertRaisesRegex(ValueError, "ID"):
            read_records(self.settings)

    def test_duplicate_image_rejected(self):
        self.labels["train"]["images"].append(copy.deepcopy(self.labels["train"]["images"][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "ID"):
            read_records(self.settings)

    def test_orphan_annotation_rejected(self):
        self.ann()["image_id"] = 999
        self.save()
        with self.assertRaises(ValueError):
            read_records(self.settings)

    def test_wrong_split_rejected(self):
        self.rows[0]["split"] = "validation"
        self.save()
        with self.assertRaisesRegex(ValueError, "분할"):
            read_records(self.settings)

    def test_manifest_count_rejected(self):
        self.rows[0]["annotation_count"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "개수"):
            read_records(self.settings)

    def test_missing_image_rejected(self):
        resolve_image_path(self.rows[0]["image_path"], self.settings["image_roots"]).unlink()
        with self.assertRaises(FileNotFoundError):
            read_records(self.settings)

    def test_bad_boxes_rejected(self):
        for box in ([0, 0, 33, 10], [-1, 2, 10, 20], [0, 0, 0, 10],
                    [0, 0, float("nan"), 10], [True, 2, 10, 20]):
            with self.subTest(box=box):
                self.ann()["bbox"] = box
                self.save()
                with self.assertRaises(ValueError):
                    read_records(self.settings)

    def test_area_mismatch_rejected(self):
        self.ann()["area"] = 199
        self.save()
        with self.assertRaisesRegex(ValueError, "면적"):
            read_records(self.settings)

    def test_ignore_and_crowd_rejected(self):
        for field in ("ignore", "iscrowd"):
            with self.subTest(field=field):
                self.ann()[field] = 1
                self.save()
                with self.assertRaises(ValueError):
                    read_records(self.settings)
                self.ann()[field] = 0

    def test_group_overlap_rejected(self):
        old = self.rows[0]["group_id"]
        name = f"select/images/{old}_0_2_0_2_90_000_200.png"
        self.rows[2].update(group_id=old, image_path=name)
        self.labels["validation"]["images"][0]["file_name"] = name
        write_png(resolve_image_path(name, self.settings["image_roots"]))
        self.save()
        with self.assertRaisesRegex(ValueError, "그룹 중복"):
            read_records(self.settings)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            resolve_image_path("select/images/../../private.png", self.settings["image_roots"])

    def test_read_does_not_modify_sources(self):
        before = {p: p.read_bytes() for p in self.package.rglob("*") if p.is_file()}
        first = read_records(self.settings)
        self.assertEqual(first, read_records(self.settings))
        self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))

    def run_cli(self, *arguments):
        custom = self.root / "settings.json"
        custom.write_text(json.dumps(self.settings))
        report = self.root / "result.json"
        process = subprocess.run([sys.executable, "-m", "detector_data", *arguments,
                                  "--report", str(report)], cwd=REPOSITORY_ROOT,
                                 env=dict(os.environ, DETECTOR_DATA_SETTINGS=str(custom)),
                                 capture_output=True, text=True)
        return process, json.loads(report.read_text())

    def test_check_command_and_report(self):
        process, report = self.run_cli("check")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["pixel_read_all_images"])

    def test_check_failure_is_recorded(self):
        self.ann()["area"] = 199
        self.save()
        process, report = self.run_cli("check")
        self.assertEqual(process.returncode, 1)
        self.assertEqual(report["status"], "failed")

    def bundle(self):
        manifest = {
            "version": "pill-smoke-package-v1", "code_sha256": {"old_code.py": "old"},
            "selected_image_ids": {"train": [1, 2], "validation": [3, 4]},
            "augmentation": DEFAULT_SETTINGS["augmentation"], "batch_size": 2, "seed": 42,
            "counts": {split: {"images": 2, "boxes": 4} for split in ("train", "validation")},
            "files_sha256": {p.relative_to(self.package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in self.package.rglob("*") if p.is_file()},
        }
        keys = ("version", "files_sha256", "code_sha256", "selected_image_ids", "augmentation", "batch_size", "seed")
        manifest["package_hash"] = hashlib.sha256(json.dumps(
            {k: manifest[k] for k in keys}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (self.package / "smoke_package_manifest.json").write_text(json.dumps(manifest))

    def test_existing_bundle_accepts_changed_code_and_keeps_data_checks(self):
        self.bundle()
        _, manifest, summary = verify_samples(self.package)
        self.assertEqual(manifest["code_sha256"], {"old_code.py": "old"})
        self.assertEqual(summary["num_train_images"], 2)

    def test_modified_bundle_image_rejected(self):
        self.bundle()
        resolve_image_path(self.rows[0]["image_path"], self.settings["image_roots"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "변경"):
            verify_samples(self.package)

    def test_modified_bundle_manifest_rejected(self):
        self.bundle()
        path = self.package / "smoke_package_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["selected_image_ids"]["train"] = [99, 100]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "식별값"):
            verify_samples(self.package)

    def test_sample_check_does_not_execute_model(self):
        self.bundle()
        process, report = self.run_cli("smoke-check", "--package", str(self.package))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertFalse(report["model_executed"])


if __name__ == "__main__":
    unittest.main()
