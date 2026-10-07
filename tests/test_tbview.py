import json
import socket
import urllib.request

import pytest

from tensorboard_book import tbview
from tensorboard_book.cli import main
from tensorboard_book.demo import make_demo


def test_link_runs_keeps_names_and_nesting(tmp_path):
    a = tmp_path / "src" / "a"
    b = tmp_path / "src" / "sweep" / "b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    dest = tmp_path / "links"
    dest.mkdir()
    tbview.link_runs({"a": a, "sweep/b": b}, dest)
    assert (dest / "a").resolve() == a.resolve()
    assert (dest / "sweep" / "b").resolve() == b.resolve()


def test_unique_names_resolve_clashes(tmp_path):
    names = tbview.unique_names(
        [tmp_path / "x" / "run", tmp_path / "y" / "run", tmp_path / "z"]
    )
    assert list(names) == ["run", "run-2", "z"]


def test_free_port_skips_busy_ports():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert tbview.free_port(start=port) != port


def test_start_shows_only_selected_runs_and_stops(tmp_path):
    root = make_demo(tmp_path / "runs")
    runs = {
        "resnet_lr3e-4": root / "resnet_lr3e-4",
        "aug_none_seed1": root / "aug_none_seed1",
    }
    inst = tbview.start(runs)
    try:
        assert tbview.wait_ready(inst), inst.log_tail()
        seen = json.load(urllib.request.urlopen(inst.url + "data/runs"))
        assert sorted(seen) == sorted(runs)
        assert tbview.start(dict(reversed(runs.items()))).id == inst.id
    finally:
        tbview.stop(inst.id)
    assert tbview.running() == []
    assert not inst.logdir.parent.exists()


def test_start_respects_the_limit(tmp_path):
    root = make_demo(tmp_path / "runs")
    first = tbview.start({"a": root / "resnet_lr3e-4"}, limit=1)
    try:
        with pytest.raises(RuntimeError):
            tbview.start({"b": root / "resnet_lr1e-4"}, limit=1)
    finally:
        tbview.stop(first.id)


def test_view_command_checks_folders(capsys):
    assert main(["view", "--logdir", "/definitely/not/here"]) == 2
    assert "not a folder" in capsys.readouterr().err
