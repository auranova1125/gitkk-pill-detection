"""공유 패키지 생성·무결성·독립 실행 경로 검사. PyTorch는 사용하지 않는다."""

import csv
import json
import shutil
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

from detector_data.detector_data_smoke_test.smoke_bundle import export_bundle, source_hashes, verify_bundle


def png(path):
    def chunk(kind, body):
        return struct.pack("!I", len(body)) + kind + body + struct.pack("!I", zlib.crc32(kind + body))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = zlib.compress((b"\0" + bytes([200, 200, 200]) * 32) * 32)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 32, 32, 8, 2, 0, 0, 0)) + chunk(b"IDAT", data) + chunk(b"IEND", b""))


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        for folder in ("detector", "detector_data", "detector_data/detector_data_smoke_test"):
            (self.repo / folder).mkdir(parents=True)
            (self.repo / folder / "__init__.py").write_text("# test fixture\n")
        self.package = self.root / "source"
        (self.package / "annotations").mkdir(parents=True)
        self.settings = {
            "package_dir": str(self.package),
            "image_roots": {"select": str(self.root / "select_aihub"), "codeit": str(self.root / "select_codeit")},
            "augmentation": {name: 0.0 for name in ("horizontal_flip", "vertical_flip", "brightness", "contrast")},
        }
        rows, selected = [], {}
        for split, offset in (("train", 0), ("validation", 2)):
            images, annotations, selected[split] = [], [], []
            for j, dataset in enumerate(("select", "codeit"), start=1):
                iid = offset + j
                group = f"K-{iid * 2 - 1:06d}-{iid * 2:06d}"
                filename = f"{group}_0_2_0_2_70_000_200.png"
                path = f"{dataset}/images/{filename}"
                png(Path(self.settings["image_roots"][dataset]) / filename)
                images.append({"id": iid, "width": 32, "height": 32, "file_name": path})
                for k in range(2):
                    annotations.append({"id": iid * 10 + k, "image_id": iid, "bbox": [k * 12, 2, 10, 20], "area": 200, "category_id": iid * 2 - 1 + k})
                rows.append({"image_id": iid, "dataset": dataset, "image_path": path, "original_image_id": iid,
                             "annotation_count": 2, "group_id": group, "split": split})
                selected[split].append({"image_id": iid, "dataset": dataset, "image_path": path, "group_id": group})
            (self.package / "annotations" / f"{split}.json").write_text(json.dumps({"images": images, "annotations": annotations, "categories": []}))
        with (self.package / "split_manifest.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        self.report = self.root / "model_smoke_report.json"
        self.record = {"command": "smoke", "status": "passed", "input_validation": "passed", "batch_size": 2,
                       "model_step": {"status": "passed"}, "source_file_sha256": source_hashes(self.package),
                       "data_summary": {"train_augmentation": self.settings["augmentation"], "num_train_instances": 4, "num_val_instances": 4},
                       "selected_images": selected}
        self.report.write_text(json.dumps(self.record))
        self.archive = self.root / "shared.zip"

    def export(self):
        return export_bundle(settings=self.settings, report_path=self.report, output_path=self.archive, repository_root=self.repo)

    def unpack(self):
        self.export()
        with zipfile.ZipFile(self.archive) as archive:
            archive.extractall(self.root / "received")
        return self.root / "received/smoke_test"

    def test_exports_exact_four_images_all_boxes_and_no_source_changes(self):
        before = source_hashes(self.package)
        bundle = self.unpack()
        manifest, _ = verify_bundle(bundle, repository_root=self.repo)
        self.assertEqual(manifest["selected_image_ids"], {"train": [1, 2], "validation": [3, 4]})
        self.assertEqual(len(list((bundle / "images").rglob("*.png"))), 4)
        self.assertEqual(manifest["counts"]["train"]["boxes"], 4)
        self.assertEqual(source_hashes(self.package), before)

    def test_replay_package_needs_no_full_dataset(self):
        bundle = self.unpack()
        shutil.rmtree(self.package)
        for folder in self.settings["image_roots"].values():
            shutil.rmtree(folder)
        _, settings = verify_bundle(bundle, repository_root=self.repo)
        self.assertEqual(Path(settings["package_dir"]), bundle)

    def test_tampered_image_rejected(self):
        bundle = self.unpack()
        next((bundle / "images").rglob("*.png")).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "변경"):
            verify_bundle(bundle, repository_root=self.repo)

    def test_different_code_rejected(self):
        bundle = self.unpack()
        (self.repo / "detector/__init__.py").write_text("# changed code\n")
        with self.assertRaisesRegex(ValueError, "코드"):
            verify_bundle(bundle, repository_root=self.repo)

    def test_windows_line_endings_are_accepted(self):
        bundle = self.unpack()
        path = self.repo / "detector/__init__.py"
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        verify_bundle(bundle, repository_root=self.repo)

    def test_changed_source_labels_require_new_smoke_run(self):
        path = self.package / "annotations/train.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "라벨"):
            self.export()

    def test_failed_smoke_report_is_not_exported(self):
        self.record["status"] = "failed"
        self.report.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, "통과"):
            self.export()

    def test_existing_shared_zip_not_overwritten(self):
        self.export()
        before = self.archive.read_bytes()
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(self.archive.read_bytes(), before)

    def test_new_images_are_not_silently_added(self):
        bundle = self.unpack()
        png(bundle / "images/select_aihub/extra.png")
        with self.assertRaisesRegex(ValueError, "구성"):
            verify_bundle(bundle, repository_root=self.repo)


if __name__ == "__main__":
    unittest.main()
