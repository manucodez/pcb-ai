from pathlib import Path

import pytest
import yaml

from scripts.check_classes import read_classes
from scripts.merge_datasets import verify_class_order, remap_label_lines, merge_dataset
import scripts.merge_datasets as merge_datasets_module


def _write_yaml(path: Path, names):
    path.write_text(yaml.dump({"names": names}, sort_keys=False))


class TestReadClasses:
    def test_list_names_unaffected(self, tmp_path):
        yaml_path = tmp_path / "data.yaml"
        _write_yaml(yaml_path, ["a", "b", "c"])
        assert read_classes(yaml_path) == ["a", "b", "c"]

    def test_int_keyed_dict_sorted_correctly(self, tmp_path):
        yaml_path = tmp_path / "data.yaml"
        _write_yaml(yaml_path, {0: "a", 1: "b", 2: "c"})
        assert read_classes(yaml_path) == ["a", "b", "c"]

    def test_string_keyed_dict_sorted_numerically_not_lexically(self, tmp_path):
        # Bug this guards against: sorted(["0","1","10","2"]) lexically
        # gives ["0","1","10","2"] — wrong order. Sorting by int(key)
        # must give 0,1,2,...,10.
        yaml_path = tmp_path / "data.yaml"
        names = {str(i): f"class_{i}" for i in range(12)}
        yaml_path.write_text(
            "names:\n" + "\n".join(f"  '{k}': {v}" for k, v in names.items())
        )
        result = read_classes(yaml_path)
        assert result == [f"class_{i}" for i in range(12)]

    def test_missing_names_key_raises(self, tmp_path):
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(yaml.dump({"train": "x"}))
        with pytest.raises(ValueError):
            read_classes(yaml_path)


class TestVerifyClassOrder:
    def test_matching_order_passes(self, tmp_path, capsys):
        dataset = tmp_path / "ds"
        dataset.mkdir()
        _write_yaml(dataset / "data.yaml", ["copper", "mousebite", "open"])
        verify_class_order(dataset, ["copper", "mousebite", "open"], "TestDS")
        assert "verified" in capsys.readouterr().out.lower()

    def test_mismatched_order_raises(self, tmp_path):
        dataset = tmp_path / "ds"
        dataset.mkdir()
        _write_yaml(dataset / "data.yaml", ["open", "copper", "mousebite"])
        with pytest.raises(ValueError):
            verify_class_order(dataset, ["copper", "mousebite", "open"], "TestDS")

    def test_missing_data_yaml_raises(self, tmp_path):
        dataset = tmp_path / "ds"
        dataset.mkdir()
        with pytest.raises(FileNotFoundError):
            verify_class_order(dataset, ["copper"], "TestDS")


class TestRemapLabelLines:
    def test_remaps_class_ids(self, tmp_path):
        lbl = tmp_path / "a.txt"
        lbl.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.05 0.05\n")
        lines = remap_label_lines(lbl, {0: 5, 1: 1})
        assert lines == ["5 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.05 0.05"]

    def test_unmapped_class_id_raises(self, tmp_path):
        lbl = tmp_path / "a.txt"
        lbl.write_text("9 0.5 0.5 0.1 0.1\n")
        with pytest.raises(ValueError):
            remap_label_lines(lbl, {0: 0})

    def test_empty_file_returns_empty_list(self, tmp_path):
        lbl = tmp_path / "a.txt"
        lbl.write_text("")
        assert remap_label_lines(lbl, {0: 0}) == []


class TestMergeDatasetSkipsUnlabeled:
    def _build_source_dataset(self, tmp_path, name="src"):
        ds = tmp_path / name
        for split in ["train", "valid", "test"]:
            (ds / split / "images").mkdir(parents=True)
            (ds / split / "labels").mkdir(parents=True)

        # One labeled image, one unlabeled image, in train split.
        (ds / "train" / "images" / "labeled.jpg").write_bytes(b"fake-jpeg-bytes")
        (ds / "train" / "images" / "unlabeled.jpg").write_bytes(b"fake-jpeg-bytes")
        (ds / "train" / "labels" / "labeled.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        return ds

    def test_skips_unlabeled_images_by_default(self, tmp_path, monkeypatch):
        ds = self._build_source_dataset(tmp_path)
        output = tmp_path / "combined"
        monkeypatch.setattr(merge_datasets_module, "OUTPUT", output)
        merge_datasets_module.make_dirs()

        from collections import Counter
        merge_dataset(ds, "src", {0: 0}, include_unlabeled=False, class_counts=Counter())

        out_images = list((output / "train" / "images").glob("*"))
        assert len(out_images) == 1
        assert "labeled" in out_images[0].name

    def test_includes_unlabeled_when_flag_set(self, tmp_path, monkeypatch):
        ds = self._build_source_dataset(tmp_path)
        output = tmp_path / "combined"
        monkeypatch.setattr(merge_datasets_module, "OUTPUT", output)
        merge_datasets_module.make_dirs()

        from collections import Counter
        merge_dataset(ds, "src", {0: 0}, include_unlabeled=True, class_counts=Counter())

        out_images = list((output / "train" / "images").glob("*"))
        assert len(out_images) == 2
