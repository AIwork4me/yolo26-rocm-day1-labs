"""Offline regressions for the X-AnyLabeling manual export workflow."""
from pathlib import Path

import pytest

from course.annotation_import import import_anylabeling_yolo
from course.lab_utils import inspect_dataset, sha256
from tests.test_lab_utils import make_dataset


def make_export(tmp_path, n=30):
    raw = make_dataset(tmp_path, n=n)
    images = raw / "images/train2017"
    labels = raw / "labels/train2017"
    classes = tmp_path / "classes.txt"
    classes.write_text("person\ncar\nchair\n", encoding="utf-8")
    for file in labels.glob("*.txt"):
        lines = []
        for row in file.read_text().splitlines():
            parts = row.split()
            parts[0] = {"0": "0", "2": "1", "56": "2"}[parts[0]]
            lines.append(" ".join(parts))
        file.write_text("\n".join(lines) + "\n")
    return images, labels, classes


def test_import_preserves_sources(tmp_path):
    images, labels, classes = make_export(tmp_path)
    before = {p.name: sha256(p) for p in labels.glob("*.txt")}
    dest = tmp_path / "student"
    report = import_anylabeling_yolo(images, labels, classes, dest)
    assert report["images"] == 30
    qa = inspect_dataset(dest / "data.yaml")
    assert qa["pass"], qa["problems"]
    assert sum(x["images"] for x in qa["splits"].values()) == 30
    assert {p.name: sha256(p) for p in labels.glob("*.txt")} == before
    with pytest.raises(FileExistsError):
        import_anylabeling_yolo(images, labels, classes, dest)


def test_rejects_invalid_and_missing_labels(tmp_path):
    images, labels, classes = make_export(tmp_path)
    one = next(labels.glob("*.txt"))
    one.write_text("3 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="Invalid YOLO"):
        import_anylabeling_yolo(images, labels, classes, tmp_path / "student")
    # Beginner hint: the error must tell the student the valid class-id range
    # and the label format, not just "invalid class id 3" (release-closure P1).
    with pytest.raises(ValueError, match=r"类别 ID 必须是 0 到 2 的整数.*0–1 的归一化小数") as excinfo:
        import_anylabeling_yolo(images, labels, classes, tmp_path / "student2")
    assert "classes.txt 共 3 行" in str(excinfo.value)
    assert not (tmp_path / "student").exists()
    assert not (tmp_path / "student2").exists()
    one.unlink()
    with pytest.raises(ValueError, match="Missing labels"):
        import_anylabeling_yolo(images, labels, classes, tmp_path / "student")
    one.write_text("0 0.5 0.5 0.2 0.2\n")  # restore a valid label for the scene test below
    # scene CSV coverage gaps must be rejected with the exact missing/extra lists
    scene = tmp_path / "scenes.csv"
    all_images = sorted(images.glob("*.png"))
    scene.write_text("image,scene\n" + "".join(
        f"{p.name},scene{i//3}\n" for i, p in enumerate(all_images)
    ), encoding="utf-8")
    scene.write_text("\n".join(scene.read_text().splitlines()[:-1]) + "\n")  # drop last row
    with pytest.raises(ValueError, match="Scene CSV must cover every image exactly once.*missing="):
        import_anylabeling_yolo(images, labels, classes, tmp_path / "student",
                                scene_csv=scene)
    assert not (tmp_path / "student").exists()


def test_grouped_scenes_never_cross_splits(tmp_path):
    images, labels, classes = make_export(tmp_path)
    scene = tmp_path / "scenes.csv"
    all_images = sorted(images.glob("*.png"))
    scene.write_text("image,scene\n" + "".join(
        f"{p.name},scene{i//3}\n" for i, p in enumerate(all_images)
    ), encoding="utf-8")
    dest = tmp_path / "student"
    import_anylabeling_yolo(images, labels, classes, dest, scene_csv=scene)
    import csv
    with (dest / "split_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for row in rows:
        groups.setdefault(row["scene"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())


def test_blank_label_lines_do_not_inflate_box_count(tmp_path):
    """Blank lines are not boxes: the manifest must agree with the dataset QA.

    Exported YOLO .txt files often carry leading/trailing blank lines (editors,
    X-AnyLabeling, manual merges). Both `label_errors` and `read_labels` skip
    blank lines, so `dataset_manifest.json["boxes"]` must use the same rule;
    otherwise the report contradicts the QA result for the very same dataset.
    """
    images, labels, classes = make_export(tmp_path)
    for path in sorted(labels.glob("*.txt"))[:5]:
        text = path.read_text(encoding="utf-8")
        path.write_text("\n" + text + "\n\n", encoding="utf-8")
    dest = tmp_path / "student"
    report = import_anylabeling_yolo(images, labels, classes, dest)
    qa = inspect_dataset(dest / "data.yaml")
    assert qa["pass"], qa["problems"]
    assert report["boxes"] == sum(x["boxes"] for x in qa["splits"].values())


def test_image_named_classes_keeps_its_label(tmp_path):
    """A photo called classes.jpg must not lose its legitimate classes.txt.

    Only the category file itself may be skipped. Excluding every .txt with
    that *name* silently drops the label of an image whose stem is `classes`,
    producing a misleading "Missing labels" error.
    """
    images, labels, classes = make_export(tmp_path)
    (images / "sample_0000.png").rename(images / "classes.png")
    (labels / "sample_0000.txt").rename(labels / "classes.txt")
    dest = tmp_path / "student"
    report = import_anylabeling_yolo(images, labels, classes, dest)
    assert report["images"] == 30
    copied = list((dest / "labels").rglob("classes.txt"))
    assert len(copied) == 1
    assert copied[0].read_text(encoding="utf-8") == (labels / "classes.txt").read_text(encoding="utf-8")


def test_classes_file_inside_labels_dir_is_excluded(tmp_path):
    """The category file passed as an argument is never read as a label file."""
    images, labels, classes = make_export(tmp_path)
    inner = labels / "classes.txt"
    inner.write_text(classes.read_text(encoding="utf-8"), encoding="utf-8")
    classes.unlink()
    dest = tmp_path / "student"
    report = import_anylabeling_yolo(images, labels, inner, dest)
    assert report["images"] == 30


def test_stray_classes_copy_in_labels_dir_is_ignored(tmp_path):
    """A stray copy of the category list in labels/ is not an orphan label."""
    images, labels, classes = make_export(tmp_path)
    (labels / "classes.txt").write_text(classes.read_text(encoding="utf-8"), encoding="utf-8")
    dest = tmp_path / "student"
    report = import_anylabeling_yolo(images, labels, classes, dest)
    assert report["images"] == 30
