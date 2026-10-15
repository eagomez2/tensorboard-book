"""Smoke test: every view of the app draws on the demo runs without errors."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tensorboard_book import db, demo, scanner

APP = Path(__file__).parents[1] / "src" / "tensorboard_book" / "app.py"
VIEWS = [
    "Experiments",
    "Run details",
    "Compare",
    "Curves",
    "Timeline",
    "Collections",
    "Fields and settings",
    "Help",
]


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    root = demo.make_demo(tmp_path_factory.mktemp("demo") / "runs")
    conn = db.connect(db.default_path(root))
    scanner.scan(conn, root, 1)
    conn.close()
    return root


@pytest.mark.parametrize("view", VIEWS)
def test_view_draws(view, runs, monkeypatch):
    monkeypatch.setenv("TBOOK_ROOT", str(runs))
    monkeypatch.setenv("TBOOK_NO_AUTH", "1")
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["view"] = view
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_default_view_opens_with_its_columns(runs, monkeypatch):
    monkeypatch.setenv("TBOOK_ROOT", str(runs))
    monkeypatch.setenv("TBOOK_NO_AUTH", "1")
    view = {
        "filters": {"starred": True},
        "scalars": ["val/acc"],
        "hparams": [],
        "show": ["status", "start"],
        "order": ["s:val/acc", "i:start", "i:status"],
        "value": "Last",
        "filter": "",
    }
    conn = db.connect(db.default_path(runs))
    db.save_settings(
        conn,
        {
            "experiment_views": {"Mine": view},
            "default_experiment_view": "Mine",
        },
    )
    try:
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert at.session_state["value_mode"] == "Last"
        assert at.session_state["filters"]["starred"] is True
        assert at.session_state["info_cols"] == ["status", "start"]
        assert at.session_state["exp_order"] == [
            "s:val/acc",
            "i:start",
            "i:status",
        ]
    finally:
        db.save_settings(
            conn,
            {"experiment_views": {}, "default_experiment_view": ""},
        )
        conn.close()


def test_curves_start_with_the_selection_and_show_labels(runs, monkeypatch):
    monkeypatch.setenv("TBOOK_ROOT", str(runs))
    monkeypatch.setenv("TBOOK_NO_AUTH", "1")
    conn = db.connect(db.default_path(runs))
    ids = [r[0] for r in conn.execute("SELECT id FROM runs ORDER BY id")]
    conn.close()
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["view"] = "Curves"
    at.session_state["selected"] = ids[:3]
    at.session_state["curve_labels_kept"] = ["tags"]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["curve_runs"] == ids[:3]
    assert at.session_state["curve_labels"] == ["tags"]

    # Back from another view with the same selection: Curves keeps the
    # runs it showed last.
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["view"] = "Curves"
    at.session_state["selected"] = ids[:3]
    at.session_state["curve_seeded"] = ids[:3]
    at.session_state["curve_runs_kept"] = ids[3:5]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["curve_runs"] == ids[3:5]
