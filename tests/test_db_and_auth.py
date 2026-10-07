import json

import pytest

from tensorboard_book import auth, db, scanner
from tensorboard_book.demo import make_demo


def test_password_hash_roundtrip():
    stored = auth.hash_password("correct horse")
    assert auth.check_hash_format(stored)
    assert auth.verify_password("correct horse", stored)
    assert not auth.verify_password("wrong", stored)
    assert not auth.verify_password("x", "not-a-hash")
    assert auth.hash_password("same") != auth.hash_password("same")  # salted


def test_annotations_survive_a_new_database(tmp_path):
    root = make_demo(tmp_path / "runs")
    db_path = db.default_path(root)
    backup_path = db.annotations_file(db_path)
    conn = db.connect(db_path)
    scanner.scan(conn, root)
    db.set_meta(conn, "annotations_path", str(backup_path))
    db.import_annotations(conn, json.loads(backup_path.read_text()))
    rid = conn.execute(
        "SELECT id FROM runs WHERE path='aug_none_seed2'"
    ).fetchone()[0]
    db.add_tags(conn, [rid], ["checked"])
    db.set_run_flag(conn, [rid], "notes", "hello")
    db.set_directions(conn, {"train/lr": "min"})
    conn.close()

    backup = json.loads(backup_path.read_text())
    db_path.unlink()
    fresh = db.connect(db_path)
    scanner.scan(fresh, root)
    result = db.import_annotations(fresh, backup)
    assert result["skipped"] == 0
    groups = {g["name"]: g for g in db.load_groups(fresh)}
    assert set(groups) == {"lr_sweep", "aug_ablation", "no_dropout_abl"}
    assert len(groups["aug_ablation"]["members"]) == 6
    runs = {r["path"]: r for r in db.load_runs(fresh).iter_rows(named=True)}
    assert runs["aug_none_seed2"]["tags"] == ["checked"]
    assert runs["aug_none_seed2"]["notes"] == "hello"
    assert db.load_directions(fresh) == {"train/lr": "min"}


def test_group_editing(tmp_path):
    conn = db.connect(tmp_path / "g.db")
    conn.execute("INSERT INTO runs(fingerprint, path) VALUES ('a', 'run_a')")
    conn.execute("INSERT INTO runs(fingerprint, path) VALUES ('b', 'run_b')")
    gid = db.create_group(conn, "ablation")
    assert db.create_group(conn, "ablation") == gid
    db.add_to_group(conn, gid, [1, 2])
    db.update_group(conn, gid, selection_metric="val/acc", columns=["val/acc"])
    db.remove_from_group(conn, gid, [2])
    group = next(g for g in db.load_groups(conn) if g["id"] == gid)
    assert group["members"] == [1]
    assert group["columns"] == ["val/acc"]
    db.rename_tag(conn, "x", "y")  # no-op on missing tag
    db.delete_group(conn, gid)
    assert db.load_groups(conn) == []


def test_properties_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "p.db")
    conn.execute("INSERT INTO runs(fingerprint, path) VALUES ('a', 'run_a')")
    conn.execute("INSERT INTO runs(fingerprint, path) VALUES ('b', 'run_b')")
    db.create_property(conn, "gpu_hours", "number", "Measured")
    db.create_property(conn, "dataset", "category")
    db.set_property(conn, [1, 2], "gpu_hours", "12.5")
    db.set_property(conn, [1], "dataset", " imagenet ")
    assert db.load_run_properties(conn) == {
        1: {"gpu_hours": 12.5, "dataset": "imagenet"},
        2: {"gpu_hours": 12.5},
    }
    with pytest.raises(ValueError):
        db.set_property(conn, [1], "gpu_hours", "lots")
    with pytest.raises(ValueError):
        db.create_property(conn, "dataset", "number")
    db.update_property(conn, "dataset", new_name="data")
    assert db.load_run_properties(conn)[1]["data"] == "imagenet"
    db.set_property(conn, [2], "gpu_hours", "")  # clears
    assert "gpu_hours" not in db.load_run_properties(conn).get(2, {})

    exported = db.export_annotations(conn)
    fresh = db.connect(tmp_path / "fresh.db")
    fresh.execute("INSERT INTO runs(fingerprint, path) VALUES ('a', 'x')")
    db.import_annotations(fresh, exported)
    props = {p["name"]: p for p in db.load_properties(fresh)}
    assert props["gpu_hours"]["kind"] == "number"
    assert db.load_run_properties(fresh) == {
        1: {"gpu_hours": 12.5, "data": "imagenet"}
    }
    db.delete_property(fresh, "data")
    assert db.load_run_properties(fresh) == {1: {"gpu_hours": 12.5}}


def test_storage_lives_in_hidden_folder_and_migrates(tmp_path):
    root = make_demo(tmp_path / "runs")
    folder = root / db.DATA_DIR
    assert db.default_path(root) == folder / db.DB_NAME
    assert (folder / db.ANNOTATIONS_NAME).is_file()  # the demo's backup
    # Files from older versions, directly in the runs folder, are moved in.
    (folder / db.ANNOTATIONS_NAME).rename(root / "tbook_annotations.json")
    (root / ".tbook.db").write_bytes(b"")
    assert db.storage_dir(root) == folder
    assert not (root / "tbook_annotations.json").exists()
    assert not (root / ".tbook.db").exists()
    assert (folder / db.ANNOTATIONS_NAME).is_file()
    assert (folder / db.DB_NAME).is_file()
    # The folder is never taken for a run, even with event files inside.
    (folder / "events.out.tfevents.1.host").write_bytes(b"")
    conn = db.connect(folder / "other.db")
    scanner.scan(conn, root)
    paths = [r[0] for r in conn.execute("SELECT path FROM runs")]
    assert paths and not any(db.DATA_DIR in p for p in paths)


def test_directions_set_and_clear(tmp_path):
    conn = db.connect(tmp_path / "x.db")
    db.set_directions(conn, {"val/acc": "max", "val/loss": "min"})
    db.clear_directions(conn, ["val/acc"])
    assert db.load_directions(conn) == {"val/loss": "min"}
    db.clear_directions(conn)
    assert db.load_directions(conn) == {}
