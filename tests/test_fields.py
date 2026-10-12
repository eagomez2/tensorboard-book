import sqlite3

import pytest

from tensorboard_book import db, fields
from tensorboard_book.fields import Field


def test_coerce_each_type():
    number = Field("n", type="number")
    assert fields.coerce(number, "4.2G") == 4.2e9
    assert fields.coerce(number, " 3 ") == 3.0
    assert fields.coerce(number, "") is None
    with pytest.raises(ValueError, match="n:"):
        fields.coerce(number, "six")
    assert fields.coerce(Field("t", type="tags"), "a, b, a") == ["a", "b"]
    assert fields.coerce(Field("b", type="bool"), "yes") is True
    assert fields.coerce(Field("d", type="date"), "2026-10-09T10:00") == (
        "2026-10-09"
    )
    with pytest.raises(ValueError):
        fields.coerce(Field("d", type="date"), "soon")


def test_labels_and_formatting():
    flops = Field(
        "FLOPs",
        group="cost",
        type="number",
        unit="FLOPs",
        scale="G",
        decimals=2,
        direction="lower",
    )
    assert flops.label() == "FLOPs (GFLOPs) ↓"
    assert flops.label(with_group=True) == "cost · FLOPs (GFLOPs) ↓"
    assert flops.format(4213000000) == "4.21"
    assert Field("x", type="number").format(1 / 3) == "0.3333"
    choice = Field("d", type="choice", options=("a", "b"))
    assert choice.option_colors() == {"a": "blue", "b": "green"}


def test_convert_between_types():
    category = Field("c", type="choice")
    assert fields.convert(category, "number", "12") == 12.0
    with pytest.raises(ValueError):
        fields.convert(category, "number", "imagenet-1k")
    tags = Field("t", type="tags")
    assert fields.convert(tags, "choice", ["a"]) == "a"
    with pytest.raises(ValueError):
        fields.convert(tags, "choice", ["a", "b"])
    assert fields.convert(Field("n", type="number"), "text", 3.0) == "3"


def test_type_change_converts_or_refuses(tmp_path):
    conn = db.connect(tmp_path / "f.db")
    conn.execute("INSERT INTO runs(fingerprint, path) VALUES ('a', 'run_a')")
    conn.execute("INSERT INTO runs(fingerprint, path) VALUES ('b', 'run_b')")
    db.create_field(conn, Field("size", type="choice"))
    db.set_field(conn, [1], "size", "12")
    db.update_field(conn, "size", Field("size", type="number"))
    assert db.load_run_fields(conn) == {1: {"size": 12.0}}
    db.update_field(conn, "size", Field("size", type="choice"))
    db.set_field(conn, [2], "size", "big")
    with pytest.raises(ValueError, match="run_b"):
        db.update_field(conn, "size", Field("size", type="number"))
    assert db.get_field(conn, "size").type == "choice"


def test_options_rename_merge_delete(tmp_path):
    conn = db.connect(tmp_path / "o.db")
    for fp in "abc":
        conn.execute(
            "INSERT INTO runs(fingerprint, path) VALUES (?, ?)",
            (fp, fp),
        )
    db.create_field(conn, Field("data", type="choice"))
    db.set_field(conn, [1], "data", "imnet")
    db.set_field(conn, [2], "data", "imagenet")
    db.set_field(conn, [3], "data", "cifar")
    db.save_options(
        conn,
        "data",
        ["cifar", "imagenet", "imagenet"],
        renames={"imnet": "imagenet"},
    )
    assert db.get_field(conn, "data").options == ("cifar", "imagenet")
    assert db.option_counts(conn, "data") == {"cifar": 1, "imagenet": 2}
    db.save_options(conn, "data", ["imagenet"], deletes=["cifar"])
    assert 3 not in db.load_run_fields(conn)
    # The built-in tags field keeps its order and renames tags on runs.
    db.add_tags(conn, [1, 2], ["b"])
    db.add_tags(conn, [3], ["a"])
    assert db.tags_field(conn).options == ("a", "b")
    db.save_options(conn, fields.TAGS, ["b", "c"], renames={"a": "c"})
    assert db.tags_field(conn).options == ("b", "c")
    runs = {
        r["id"]: r["tags"] for r in db.load_runs(conn).iter_rows(named=True)
    }
    assert runs[3] == ["c"]


def test_version_1_database_is_upgraded(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        db.SCHEMA.split("CREATE TABLE IF NOT EXISTS properties")[0]
        + """
        CREATE TABLE properties (
            name TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('number', 'category')),
            description TEXT NOT NULL DEFAULT '',
            created REAL
        );
        CREATE TABLE run_properties (
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            name TEXT NOT NULL
                REFERENCES properties(name) ON DELETE CASCADE
                ON UPDATE CASCADE,
            value NOT NULL,
            PRIMARY KEY (run_id, name)
        );
        INSERT INTO runs(fingerprint, path) VALUES ('a', 'run_a');
        INSERT INTO properties VALUES ('dataset', 'category', 'Data', 0);
        INSERT INTO properties VALUES ('rating', 'number', '', 0);
        INSERT INTO run_properties VALUES (1, 'dataset', 'imagenet');
        INSERT INTO run_properties VALUES (1, 'rating', 4.0);
        """
    )
    old.commit()
    old.close()
    conn = db.connect(path)
    defined = {f.name: f for f in db.load_fields(conn)}
    assert defined["dataset"].type == "choice"
    assert defined["dataset"].options == ("imagenet",)
    assert defined["rating"].type == "number"
    assert db.load_run_fields(conn) == {
        1: {"dataset": "imagenet", "rating": 4.0}
    }
    # The foreign key still points at the rebuilt table.
    db.update_field(conn, "dataset", Field("data", type="choice"))
    assert db.load_run_fields(conn)[1]["data"] == "imagenet"
    assert db.get_meta(conn, "schema_version") == "2"


def test_version_1_annotations_import(tmp_path):
    conn = db.connect(tmp_path / "a.db")
    conn.execute("INSERT INTO runs(fingerprint, path) VALUES ('a', 'run_a')")
    db.import_annotations(
        conn,
        {
            "format": "tensorboard-book-annotations",
            "version": 1,
            "properties": [{"name": "dataset", "kind": "category"}],
            "runs": [
                {
                    "fingerprint": "a",
                    "path": "run_a",
                    "starred": False,
                    "archived": False,
                    "notes": "",
                    "tags": [],
                    "properties": {"dataset": "cifar"},
                }
            ],
        },
    )
    assert db.get_field(conn, "dataset").options == ("cifar",)
    assert db.load_run_fields(conn) == {1: {"dataset": "cifar"}}


def test_fields_file_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "x.db")
    db.create_field(conn, Field("wer", group="eval", type="number", unit="%"))
    data = db.export_fields(conn)
    other = db.connect(tmp_path / "y.db")
    assert db.import_fields(other, data)["added"] == ["wer"]
    assert db.import_fields(other, data)["kept"] == ["wer"]
    assert db.get_field(other, "wer").unit == "%"
