import shutil

import numpy as np
import polars as pl
import pytest

from tensorboard_book import db, scanner
from tensorboard_book.demo import make_demo


@pytest.fixture()
def demo(tmp_path):
    root = make_demo(tmp_path / "runs")
    conn = db.connect(db.default_path(root))
    scanner.scan(conn, root)
    return root, conn


def runs_by_path(conn):
    return {r["path"]: r for r in conn.execute("SELECT * FROM runs")}


def test_discovers_every_demo_run(demo):
    root, conn = demo
    paths = set(runs_by_path(conn))
    assert "keras_baseline" in paths
    assert "resnet_lr3e-4" in paths
    assert len(paths) == 14


def test_keras_subfolders_become_prefixed_tensor_scalars(demo):
    _, conn = demo
    rid = runs_by_path(conn)["keras_baseline"]["id"]
    tags = {
        r["tag"]
        for r in conn.execute("SELECT tag FROM metrics WHERE run_id=?", (rid,))
    }
    assert tags == {
        "train/epoch_loss",
        "train/epoch_accuracy",
        "validation/epoch_loss",
        "validation/epoch_accuracy",
    }


def test_metric_summary_and_series(demo):
    _, conn = demo
    rid = runs_by_path(conn)["resnet_lr3e-4"]["id"]
    row = conn.execute(
        "SELECT * FROM metrics WHERE run_id=? AND tag='val/acc'", (rid,)
    ).fetchone()
    series = db.load_series(conn, rid, "val/acc")
    assert row["n"] == len(series) == 20
    assert row["max"] == pytest.approx(series["value"].max())
    best = series.row(series["value"].arg_max(), named=True)
    assert row["step_max"] == int(best["step"])


def test_nan_run_is_flagged_and_best_ignores_nan(demo):
    _, conn = demo
    run = runs_by_path(conn)["resnet_lr1e-2"]
    assert run["has_nonfinite"] == 1
    row = conn.execute(
        "SELECT * FROM metrics WHERE run_id=? AND tag='train/loss'",
        (run["id"],),
    ).fetchone()
    assert row["nonfinite"] > 0
    assert np.isfinite(row["min"])


def test_resumed_run_has_two_segments(demo):
    _, conn = demo
    run = runs_by_path(conn)["resnet_lr3e-3"]
    assert run["n_segments"] == 2
    assert run["wall_span"] - run["compute_time"] == pytest.approx(
        12 * 3600, rel=0.01
    )


def test_hparams_from_events_and_lenient_yaml(demo):
    import json

    _, conn = demo
    runs = runs_by_path(conn)
    sweep = json.loads(runs["resnet_lr1e-3"]["hparams"])
    assert sweep["batch_size"] == 256 and isinstance(sweep["batch_size"], int)
    assert sweep["optimizer.weight_decay"] == 0.05
    keras = json.loads(runs["keras_baseline"]["hparams"])
    assert keras["model"] == "mobilenet_v2"  # despite the !!python/object tag


def test_text_summaries_are_extracted(demo):
    import json

    _, conn = demo
    texts = json.loads(runs_by_path(conn)["no_dropout_s1"]["texts"])
    assert "regularization" in texts["notes/text_summary"]


def test_rescan_skips_unchanged_runs(demo):
    root, conn = demo
    counts = scanner.scan(conn, root)
    assert counts["parsed"] == 0 and counts["new"] == 0


def test_rename_keeps_record_and_annotations(demo):
    root, conn = demo
    rid = runs_by_path(conn)["resnet_lr1e-3"]["id"]
    db.add_tags(conn, [rid], ["keep-me"])
    (root / "resnet_lr1e-3").rename(root / "renamed_run")
    counts = scanner.scan(conn, root)
    assert counts["renamed"] == 1 and counts["new"] == 0
    run = runs_by_path(conn)["renamed_run"]
    assert run["id"] == rid
    tags = [
        r["tag"]
        for r in conn.execute(
            "SELECT tag FROM run_tags WHERE run_id=?", (rid,)
        )
    ]
    assert tags == ["keep-me"]


def test_deleted_run_is_marked_missing(demo):
    root, conn = demo
    shutil.rmtree(root / "aug_none_seed1")
    counts = scanner.scan(conn, root)
    assert counts["missing"] == 1
    assert runs_by_path(conn)["aug_none_seed1"]["missing"] == 1


def test_copied_run_gets_its_own_record(demo):
    root, conn = demo
    original = runs_by_path(conn)["resnet_lr1e-4"]
    shutil.copytree(root / "resnet_lr1e-4", root / "resnet_lr1e-4_copy")
    counts = scanner.scan(conn, root)
    assert counts["new"] == 1
    runs = runs_by_path(conn)
    assert runs["resnet_lr1e-4"]["id"] == original["id"]
    assert runs["resnet_lr1e-4_copy"]["fingerprint"] != original["fingerprint"]


def test_appended_events_trigger_reparse(demo):
    root, conn = demo
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter

    event_file = next((root / "resnet_lr1e-4").glob("events.out.tfevents.*"))
    value = summary_pb2.Summary.Value(tag="val/acc", simple_value=0.99)
    event = event_pb2.Event(
        wall_time=2e9, step=60000, summary=summary_pb2.Summary(value=[value])
    )
    with open(event_file, "ab") as fh:
        RecordWriter(fh).write(event.SerializeToString())
    counts = scanner.scan(conn, root)
    assert counts["parsed"] == 1
    rid = runs_by_path(conn)["resnet_lr1e-4"]["id"]
    best = conn.execute(
        "SELECT max FROM metrics WHERE run_id=? AND tag='val/acc'", (rid,)
    ).fetchone()["max"]
    assert best == pytest.approx(0.99)


def test_truncated_event_file_is_read_up_to_the_cut(tmp_path):
    root = make_demo(tmp_path / "runs")
    event_file = next((root / "resnet_lr1e-4").glob("events.out.tfevents.*"))
    data = event_file.read_bytes()
    event_file.write_bytes(data[: len(data) - 7])
    conn = db.connect(tmp_path / "t.db")
    scanner.scan(conn, root)
    assert runs_by_path(conn)["resnet_lr1e-4"]["n_events"] > 100


def test_depth_two_and_worker_processes(tmp_path):
    root = tmp_path / "project"
    make_demo(root / "sweeps")
    conn = db.connect(tmp_path / "d.db")
    counts = scanner.scan(conn, root, depth=2, workers=2)
    assert counts["total"] == 14
    assert "sweeps/resnet_lr1e-3" in runs_by_path(conn)


def test_downsampling_keeps_extremes():
    values = np.sin(np.linspace(0, 20, 10_000))
    values[1234] = 5.0
    keep = scanner.downsample_indices(values, 100)
    assert len(keep) <= 104
    assert 1234 in keep and 0 in keep and 9999 in keep


def test_segments_split_on_gaps():
    walls = np.array([0, 10, 20, 5000, 5010])
    assert scanner.compute_segments(walls, 100) == [
        [0.0, 20.0],
        [5000.0, 5010.0],
    ]


def test_cli_scan_with_spawned_workers(tmp_path):
    """Regression: parallel parsing must work with "spawn" (macOS default)."""
    import json
    import subprocess
    import sys

    root = make_demo(tmp_path / "runs")
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorboard_book.cli",
            "scan",
            str(root),
            "--workers",
            "3",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    counts = json.loads(out.stdout.strip().splitlines()[-1])
    assert counts["parsed"] == 14


def test_app_never_starts_worker_processes():
    """Regression: spawned workers re-import the Streamlit script and crash."""
    from pathlib import Path

    import tensorboard_book

    source = (Path(tensorboard_book.__file__).parent / "app.py").read_text()
    assert "scanner.scan(conn, ROOT, DEPTH, 1," in source
    assert "ProcessPool" not in source


def test_changing_pause_setting_reparses_every_run(demo):
    root, conn = demo
    assert scanner.scan(conn, root)["parsed"] == 0
    db.save_settings(
        conn, {"gap_minutes": 60 * 24}
    )  # a 12 h pause is no pause
    assert scanner.scan(conn, root)["parsed"] == 14
    run = runs_by_path(conn)["resnet_lr3e-3"]
    assert run["n_segments"] == 1
    assert run["compute_time"] == pytest.approx(run["wall_span"])


def test_default_thresholds_fit_long_epochs():
    assert db.DEFAULT_SETTINGS["active_minutes"] == 300
    assert (
        db.DEFAULT_SETTINGS["gap_minutes"]
        > db.DEFAULT_SETTINGS["active_minutes"]
    )


def test_load_runs_and_details(demo):
    _, conn = demo
    runs = db.load_runs(conn)
    assert runs.height == 14
    assert runs.schema["tags"] == pl.List(pl.String)
    details = db.load_run_details(conn)
    keras = runs.filter(pl.col("path") == "keras_baseline")["id"][0]
    assert details[keras]["hparams"]["model"] == "mobilenet_v2"
    series = db.load_series(conn, keras, "train/epoch_loss")
    assert series.columns == ["step", "wall_time", "value"]
    assert db.load_series(conn, keras, "nope").is_empty()
