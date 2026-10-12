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
