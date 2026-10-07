import datetime
import socket
from pathlib import Path

import pytest

from tensorboard_book import auth, cli


def test_port_is_free_sees_a_listening_socket():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert not cli.port_is_free("127.0.0.1", port)
    assert cli.port_is_free("127.0.0.1", port)


def serve(tmp_path, monkeypatch, *extra):
    calls = []
    monkeypatch.setattr(
        cli.subprocess, "call", lambda cmd, env: calls.append((cmd, env)) or 0
    )
    args = cli.build_parser().parse_args(["serve", str(tmp_path), *extra])
    return args.func(args), calls


def test_local_launch_needs_no_password(tmp_path, monkeypatch):
    monkeypatch.delenv(auth.EDITOR_ENV, raising=False)
    code, calls = serve(tmp_path, monkeypatch)
    assert code == 0
    cmd, env = calls[0]
    assert env[auth.NO_AUTH_ENV] == "1"
    # The first free port, and no headless flag, so a browser tab opens.
    port = int(cmd[cmd.index("--server.port") + 1])
    assert 8501 <= port < 8601 and "--server.headless" not in cmd


def test_password_is_enforced_once_set(tmp_path, monkeypatch):
    monkeypatch.setenv(auth.EDITOR_ENV, auth.hash_password("pw"))
    monkeypatch.setenv(auth.NO_AUTH_ENV, "1")
    code, calls = serve(tmp_path, monkeypatch, "--no-browser", "--port", "0")
    cmd, env = calls[0]
    assert auth.NO_AUTH_ENV not in env
    assert cmd[cmd.index("--server.headless") + 1] == "true"
    assert cmd[cmd.index("--server.port") + 1] == "0"


def test_remote_host_needs_a_password(tmp_path, monkeypatch):
    monkeypatch.delenv(auth.EDITOR_ENV, raising=False)
    code, calls = serve(tmp_path, monkeypatch, "--host", "0.0.0.0")
    assert code == 2 and not calls


def test_version_shows_the_years():
    first = datetime.date(cli.FIRST_YEAR, 6, 1)
    later = datetime.date(cli.FIRST_YEAR + 2, 1, 1)
    version = cli.__version__
    assert cli.version_text(first) == (
        f"tensorboard-book version {version} {cli.FIRST_YEAR}"
    )
    assert cli.version_text(later) == (
        f"tensorboard-book version {version} "
        f"{cli.FIRST_YEAR} - {cli.FIRST_YEAR + 2}"
    )


@pytest.mark.parametrize("flag", ["-v", "--version"])
def test_version_flags(flag, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([flag])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == cli.version_text()


def test_readme_badge_shows_the_version():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert f"badge/version-{cli.__version__}-" in readme
