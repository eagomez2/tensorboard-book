"""Open a selection of runs in TensorBoard (the tensorboard-view idea).

TensorBoard shows every run below one ``--logdir``. To show only some
runs, their folders are symlinked into a temporary folder and TensorBoard
is pointed at that folder. The links keep each run's name, so TensorBoard
lists the runs under the same names as tensorboard-book.

Two ways to use it:

* ``tensorboard-book view --logdir RUN [RUN ...] [TensorBoard args]``
  runs TensorBoard in the foreground, like the original tensorboard-view.
* The app starts instances in the background with :func:`start`. They
  are listed by :func:`running`, stopped with :func:`stop`, and all of
  them stop when the app exits.

TensorBoard has no password, so instances listen on ``127.0.0.1`` unless
told otherwise; that way they never expose runs past the app's login.
"""

from __future__ import annotations

import atexit
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

LOCAL_HOST = "127.0.0.1"
FIRST_PORT = 6006


def link_runs(runs: dict[str, Path], dest: Path) -> Path:
    """Symlink run folders into ``dest`` under the given names.

    Names with slashes (runs nested below the root) become subfolders, so
    TensorBoard shows the same ``a/b`` name.

    Args:
        runs: Mapping of display name to run folder.
        dest: An empty folder to put the links in.

    Returns:
        ``dest``.
    """
    for name, path in runs.items():
        link = dest / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path(path).resolve(), target_is_directory=True)
    return dest


def unique_names(paths: list[str | Path]) -> dict[str, Path]:
    """Name folders by their base name, adding ``-2``, ``-3`` on clashes.

    Args:
        paths: Run folders given on the command line.

    Returns:
        Mapping of unique display name to folder.
    """
    names: dict[str, Path] = {}
    for path in paths:
        base = Path(path).resolve().name or "run"
        name, n = base, 1
        while name in names:
            n += 1
            name = f"{base}-{n}"
        names[name] = Path(path)
    return names


def free_port(host: str = LOCAL_HOST, start: int = FIRST_PORT) -> int:
    """Find the first port from ``start`` that nothing listens on.

    Args:
        host: Address the server will bind to.
        start: First port to try.

    Returns:
        A free port.

    Raises:
        RuntimeError: If no port in 100 tries is free.
    """
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port between {start} and {start + 99}")


def command(logdir: Path, host: str, port: int, extra: list[str]) -> list:
    """Build the TensorBoard command line.

    TensorBoard runs with this Python interpreter, so it comes from the
    same environment as tensorboard-book.

    Args:
        logdir: Folder with the linked runs.
        host: Address to listen on.
        port: Port to listen on.
        extra: Additional TensorBoard arguments.

    Returns:
        The argument list for ``subprocess``.
    """
    return [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        host,
        "--port",
        str(port),
        *extra,
    ]


def view(paths: list[str | Path], extra: list[str]) -> int:
    """Run TensorBoard in the foreground on some run folders.

    This is the ``tensorboard-book view`` command. The temporary folder is
    removed when TensorBoard exits.

    Args:
        paths: Run folders to show.
        extra: TensorBoard arguments (e.g. ``--port 6007``).

    Returns:
        TensorBoard's exit code (0 after Ctrl+C).
    """
    with tempfile.TemporaryDirectory(prefix="tbook-view-") as tmp:
        link_runs(unique_names(paths), Path(tmp))
        try:
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    tmp,
                    *extra,
                ]
            )
        except KeyboardInterrupt:
            return 0


@dataclass
class Instance:
    """A TensorBoard started by the app in the background."""

    id: int
    runs: list[str]
    host: str
    port: int
    logdir: Path
    process: subprocess.Popen
    started: float

    @property
    def url(self) -> str:
        """The address to open in a browser."""
        host = self.host
        if host in (LOCAL_HOST, "localhost", "::1"):
            host = "localhost"
        elif host in ("0.0.0.0", "::"):
            host = socket.gethostname()
        return f"http://{host}:{self.port}/"

    @property
    def alive(self) -> bool:
        """Whether the TensorBoard process is still running."""
        return self.process.poll() is None

    def log_tail(self, lines: int = 20) -> str:
        """Return the end of TensorBoard's output, to explain failures."""
        path = self.logdir.parent / "tensorboard.log"
        try:
            return "\n".join(path.read_text().splitlines()[-lines:])
        except OSError:
            return ""

    def ready(self) -> bool:
        """Whether TensorBoard accepts connections yet."""
        try:
            with socket.create_connection((self.host, self.port), 0.5):
                return True
        except OSError:
            return False


_INSTANCES: dict[int, Instance] = {}
_LOCK = threading.Lock()


def start(
    runs: dict[str, Path],
    host: str = LOCAL_HOST,
    extra: list[str] | None = None,
    limit: int = 5,
) -> Instance:
    """Start TensorBoard in the background on a selection of runs.

    If an instance showing exactly these runs is still running, it is
    returned instead of starting another one.

    Args:
        runs: Mapping of display name to run folder.
        host: Address TensorBoard listens on.
        extra: Additional TensorBoard arguments.
        limit: Maximum number of instances running at once.

    Returns:
        The instance. It may need a few seconds before :meth:`ready`.

    Raises:
        RuntimeError: If ``limit`` instances are already running.
    """
    names = sorted(runs)
    with _LOCK:
        for inst in _prune():
            if inst.runs == names:
                return inst
        if len(_INSTANCES) >= limit:
            raise RuntimeError(
                f"{limit} TensorBoards are already running. Stop one first."
            )
        workdir = Path(tempfile.mkdtemp(prefix="tbook-tensorboard-"))
        logdir = link_runs(runs, workdir / "runs")
        port = free_port(host)
        with open(workdir / "tensorboard.log", "wb") as log:
            process = subprocess.Popen(
                command(logdir, host, port, extra or []),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        inst = Instance(
            id=max(_INSTANCES, default=0) + 1,
            runs=names,
            host=host,
            port=port,
            logdir=logdir,
            process=process,
            started=time.time(),
        )
        _INSTANCES[inst.id] = inst
        return inst


def wait_ready(inst: Instance, timeout: float = 30.0) -> bool:
    """Wait until TensorBoard answers, or until it exits or times out.

    Args:
        inst: A started instance.
        timeout: Seconds to wait.

    Returns:
        True once it accepts connections.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if inst.ready():
            return True
        if not inst.alive:
            return False
        time.sleep(0.3)
    return False


def _prune() -> list[Instance]:
    """Forget instances whose process exited; return the live ones."""
    for inst_id, inst in list(_INSTANCES.items()):
        if not inst.alive:
            shutil.rmtree(inst.logdir.parent, ignore_errors=True)
            del _INSTANCES[inst_id]
    return list(_INSTANCES.values())


def running() -> list[Instance]:
    """Return the TensorBoards the app started that are still running."""
    with _LOCK:
        return _prune()


def stop(inst_id: int) -> None:
    """Stop one instance and remove its temporary folder.

    Args:
        inst_id: :attr:`Instance.id`.
    """
    with _LOCK:
        inst = _INSTANCES.pop(inst_id, None)
    if inst is None:
        return
    inst.process.terminate()
    try:
        inst.process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        inst.process.kill()
    shutil.rmtree(inst.logdir.parent, ignore_errors=True)


@atexit.register
def stop_all() -> None:
    """Stop every instance; runs automatically when the app exits."""
    for inst_id in list(_INSTANCES):
        stop(inst_id)
