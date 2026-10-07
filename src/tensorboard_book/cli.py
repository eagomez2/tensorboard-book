"""Command line entry point: ``tensorboard-book``.

Usage::

    tensorboard-book [ROOT] [--host H] [--port P] [--depth N] [--db PATH]
                     [--workers N] [--no-auth] [--no-browser]
    tensorboard-book scan ROOT [--depth N] [--db PATH] [--workers N] [--force]
    tensorboard-book hash-password
    tensorboard-book demo DEST
    tensorboard-book export-annotations ROOT [--db PATH] [-o FILE]
    tensorboard-book import-annotations ROOT FILE [--db PATH]
    tensorboard-book view --logdir RUN [RUN ...] [TensorBoard options]

Without a subcommand, ``tensorboard-book ROOT`` scans the folder and starts the
web app for it.
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import click

from tensorboard_book import __version__, auth, db, scanner, tbview

SUBCOMMANDS = {
    "serve",
    "scan",
    "hash-password",
    "demo",
    "export-annotations",
    "import-annotations",
    "view",
}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
# The year of the first release, as in LICENSE, shown by --version.
FIRST_YEAR = 2026
# Warm light and dark themes with one teal accent, the same as modelboard's.
# Users switch between them in the app menu (Settings > Theme); by default
# the app follows the system setting. Fonts ship inside the package, in
# ``static/fonts``. Options the installed Streamlit does not know are
# skipped.
SHARED_THEME = {
    "theme.font": "IBM Plex Sans, sans-serif",
    "theme.headingFont": "Newsreader, serif",
    "theme.headingFontWeights": "500",
    "theme.codeFont": "IBM Plex Mono, monospace",
    "theme.baseRadius": "0.6rem",
    "theme.showSidebarBorder": "false",
}
COLORS = {
    "light": {
        "primaryColor": "#0F5F5A",
        "backgroundColor": "#FAF9F6",
        "secondaryBackgroundColor": "#F1EFE9",
        "textColor": "#1B1A17",
        "borderColor": "#E4E1D8",
        "dataframeBorderColor": "#ECE9E1",
        "dataframeHeaderBackgroundColor": "#F6F4EF",
        "linkColor": "#0F5F5A",
    },
    "dark": {
        "primaryColor": "#2E8B80",
        "backgroundColor": "#161614",
        "secondaryBackgroundColor": "#22211E",
        "textColor": "#ECE9E1",
        "borderColor": "#34322D",
        "dataframeBorderColor": "#2C2A26",
        "dataframeHeaderBackgroundColor": "#1D1C19",
        "linkColor": "#6CC5BB",
    },
}
THEME = dict(SHARED_THEME)
for _mode, _colors in COLORS.items():
    for _key, _value in _colors.items():
        THEME[f"theme.{_mode}.{_key}"] = _value


# Terminal messages look like Streamlit's own: indented by two spaces, blue
# labels and bold values. click (which Streamlit uses for the same thing)
# drops the colors when the output isn't a terminal.


def heading(text: str) -> None:
    """Print a title between blank lines."""
    click.echo()
    click.secho(f"  {text}", fg="blue", bold=True)
    click.echo()


def field(label: str, value: str) -> None:
    """Print ``label: value`` with a blue label and a bold value."""
    click.echo(
        "  "
        + click.style(f"{label}:", fg="blue")
        + " "
        + click.style(value, bold=True)
    )


def error(message: str, *hints: str) -> None:
    """Print an error, and optional hints below it, to stderr."""
    click.echo(
        "  " + click.style("Error:", fg="red", bold=True) + f" {message}",
        err=True,
    )
    for hint in hints:
        click.echo(f"  {hint}", err=True)


def warning(message: str) -> None:
    """Print a warning to stderr."""
    click.echo(
        "  " + click.style("Warning:", fg="yellow", bold=True) + f" {message}",
        err=True,
    )


def print_progress(done: int, total: int, path: str) -> None:
    """Print a one-line scan progress indicator.

    Args:
        done: Runs parsed so far.
        total: Runs to parse.
        path: The run just parsed.
    """
    click.echo(
        "\r  "
        + click.style("Parsing:", fg="blue")
        + f" [{done}/{total}] {path[:60]:<60}",
        nl=False,
    )


def copyright_years(today: datetime.date | None = None) -> str:
    """Return the years of the project, for ``--version`` and the docs.

    Args:
        today: The date to take the current year from (defaults to today).

    Returns:
        For example ``2026 - 2027``, or a single year while it is still the
        first one.
    """
    year = (today or datetime.date.today()).year
    return f"{FIRST_YEAR} - {year}" if year > FIRST_YEAR else str(year)


def version_text(today: datetime.date | None = None) -> str:
    """Return what ``--version`` prints.

    Args:
        today: The date to take the current year from (defaults to today).

    Returns:
        For example ``tensorboard-book version 0.7.0 2026 - 2027``.
    """
    return f"tensorboard-book version {__version__} {copyright_years(today)}"


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the Streamlit app.

    Args:
        args: Parsed arguments.

    Returns:
        The Streamlit process exit code.
    """
    root = Path(args.root).resolve()
    if not root.is_dir():
        error(f"{root} is not a folder.")
        return 2
    env = dict(os.environ)
    local = args.host in LOCAL_HOSTS
    if args.no_auth and not local:
        error(
            "--no-auth only works with a local --host (127.0.0.1).",
            f"Set {auth.EDITOR_ENV} to open the app to other computers.",
        )
        return 2
    stored = env.get(auth.EDITOR_ENV, "")
    if args.no_auth or (local and not stored):
        # Only this computer can reach a local host, so the password is
        # optional there: it is asked for only once the hash is set.
        env[auth.NO_AUTH_ENV] = "1"
    else:
        env.pop(auth.NO_AUTH_ENV, None)
        if not stored:
            error(
                f"--host {args.host} opens the app to other computers, so "
                f"it needs {auth.EDITOR_ENV}.",
                "Create it with:  tensorboard-book hash-password",
                f"then:            export {auth.EDITOR_ENV}='<value>'",
            )
            return 2
        for name in (auth.EDITOR_ENV, auth.VIEWER_ENV):
            if env.get(name) and not auth.check_hash_format(env[name]):
                error(
                    f"{name} is malformed.",
                    "Create it again with:  tensorboard-book hash-password",
                )
                return 2
    if args.port is not None and not port_is_free(args.host, args.port):
        # Streamlit would stop too, but only after the scan.
        error(
            f"Port {args.port} is in use.",
            "Leave out --port to use the next free one.",
        )
        return 2
    db_path = Path(args.db).resolve() if args.db else db.default_path(root)
    heading(f"tensorboard-book {__version__}")
    field("Runs folder", str(root))
    field("Database", str(db_path))
    # The first scan runs here, where worker processes are safe, so the app
    # opens with everything already indexed.
    conn = db.connect(db_path)
    was_empty = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    counts = scanner.scan(
        conn, root, args.depth, args.workers, progress=print_progress
    )
    if counts["parsed"]:
        click.echo("\r" + " " * 90 + "\r", nl=False)  # Clear the progress.
    field(
        "Runs",
        f"{counts['total']} ({counts['new']} new, {counts['parsed']} parsed)",
    )
    annotations = db.annotations_file(db_path)
    db.set_meta(conn, "annotations_path", str(annotations))
    conn.commit()
    if was_empty and annotations.exists():
        # A fresh database: restore tags, notes and groups from the backup.
        try:
            result = db.import_annotations(
                conn, json.loads(annotations.read_text())
            )
            field(
                "Annotations",
                f"restored for {result['matched']} runs from {annotations}",
            )
        except ValueError as exc:
            warning(f"could not restore {annotations}: {exc}")
    conn.close()
    env.update(
        TBOOK_ROOT=str(root),
        TBOOK_DB=str(db_path),
        TBOOK_DEPTH=str(args.depth),
        TBOOK_TENSORBOARD_HOST=args.tensorboard_host,
        # No usage statistics, and no "what's new"/email prompts either.
        STREAMLIT_BROWSER_GATHER_USAGE_STATS="false",
        STREAMLIT_SERVER_SHOW_EMAIL_PROMPT="false",
    )
    app = Path(__file__).with_name("app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        args.host,
        "--server.fileWatcherType",
        "none",
        "--server.runOnSave",
        "false",
        # Serves only the bundled fonts in ``static/``.
        "--server.enableStaticServing",
        "true",
        "--server.enableXsrfProtection",
        "true",
        "--browser.gatherUsageStats",
        "false",
        # Only Streamlit's address block and real problems, no info logs.
        "--logger.level",
        "warning",
        # "viewer" keeps the app menu, with Settings > Theme for switching
        # between light and dark, and hides the developer options.
        "--client.toolbarMode",
        "viewer",
        *theme_flags(),
    ]
    if args.port is not None:
        # A port that was asked for is strict: Streamlit exits if it is busy.
        command += ["--server.port", str(args.port)]
    if args.no_browser:
        # Otherwise Streamlit opens a browser tab when it has a display, and
        # quietly skips it when there is none (e.g. over SSH).
        command += ["--server.headless", "true"]
    if "logger.hideWelcomeMessage" in streamlit_options():
        # Streamlit's welcome message ends with an ad for its agent skills,
        # which can only be hidden together with it, so print the same
        # message here instead. That needs the port before Streamlit starts:
        # the first free one from 8501, as Streamlit would pick.
        port = args.port or next(
            (p for p in range(8501, 8601) if port_is_free(args.host, p)), 8501
        )
        command += [
            "--server.port",
            str(port),
            "--logger.hideWelcomeMessage",
            "true",
        ]
        welcome(args.host, port)
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 0


def port_is_free(host: str, port: int) -> bool:
    """Check whether the app could listen on ``host:port``.

    Args:
        host: Address to listen on.
        port: Port to check.

    Returns:
        False if another program already listens there.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return True  # Let Streamlit report the bad address.
    family, kind, proto, _, address = infos[0]
    with socket.socket(family, kind, proto) as sock:
        if os.name != "nt":
            # As the server does, so a recently closed port counts as free.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(address)
        except OSError:
            return False
    return True


def welcome(host: str, port: int) -> None:
    """Print Streamlit's welcome message: where to open the app.

    Args:
        host: Address the app listens on.
        port: Port the app listens on.
    """
    heading("You can now view your Streamlit app in your browser.")
    if host in ("0.0.0.0", "::", ""):
        urls = [("Local URL", "localhost")]
        try:
            # Finds the address of the network interface; nothing is sent.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                urls.append(("Network URL", sock.getsockname()[0]))
        except OSError:
            pass
    else:
        urls = [("URL", f"[{host}]" if ":" in host else host)]
    for label, address in urls:
        field(label, f"http://{address}:{port}")
    click.echo()


def streamlit_options() -> set[str]:
    """Return the options the installed Streamlit knows, e.g. ``theme.font``.

    Returns:
        The option names, or an empty set if they can't be read.
    """
    try:
        from streamlit import config

        # The list of known options, read without parsing the config
        # files, so Streamlit doesn't warn that options changed.
        template = getattr(config, "_config_options_template", None)
        return set(template or config.get_config_options())
    except Exception:  # noqa: BLE001 - never block starting the app.
        return set()


def theme_flags() -> list[str]:
    """Build the theme options the installed Streamlit supports.

    Returns:
        ``--theme.*`` command line flags with their values.
    """
    known = streamlit_options()
    flags: list[str] = []
    for key, value in THEME.items():
        if key in known:
            flags += [f"--{key}", value]
    return flags


def cmd_scan(args: argparse.Namespace) -> int:
    """Index a folder without starting the app (for cron or big folders).

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    root = Path(args.root).resolve()
    conn = db.connect(
        Path(args.db).resolve() if args.db else db.default_path(root)
    )

    counts = scanner.scan(
        conn, root, args.depth, args.workers, args.force, print_progress
    )
    print()
    print(json.dumps(counts))
    return 0


def cmd_hash_password(args: argparse.Namespace) -> int:
    """Ask for a password twice and print the environment variable to set.

    Args:
        args: Parsed arguments (unused).

    Returns:
        Exit code.
    """
    password = getpass.getpass("Password: ")
    if len(password) < 8:
        error("Use at least 8 characters.")
        return 2
    if getpass.getpass("Repeat: ") != password:
        error("The passwords don't match.")
        return 2
    print(f"{auth.EDITOR_ENV}={auth.hash_password(password)}")
    print(
        (
            f"(Use the same value for {auth.VIEWER_ENV} to create a read-only "
            "password.)"
        ),
        file=sys.stderr,
    )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Create a folder of fake runs to try the app with.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from tensorboard_book.demo import make_demo

    try:
        dest = make_demo(args.dest)
    except FileExistsError as exc:
        error(str(exc))
        return 2
    print(f"Demo runs written to {dest}. Try:  tensorboard-book {dest}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Write tags, notes, groups and settings to a JSON file.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    root = Path(args.root).resolve()
    conn = db.connect(
        Path(args.db).resolve() if args.db else db.default_path(root)
    )
    data = db.export_annotations(conn, args.output)
    print(
        f"Exported {len(data['runs'])} annotated runs and "
        f"{len(data['groups'])} groups"
    )
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Merge annotations from a JSON file into a folder's database.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    root = Path(args.root).resolve()
    conn = db.connect(
        Path(args.db).resolve() if args.db else db.default_path(root)
    )
    scanner.scan(conn, root, args.depth)
    result = db.import_annotations(
        conn, json.loads(Path(args.file).read_text())
    )
    print(f"Matched {result['matched']} runs, {result['skipped']} not found")
    return 0


def cmd_view(argv: list[str]) -> int:
    """Open some run folders in TensorBoard (like tensorboard-view).

    Unknown options are passed on to TensorBoard, e.g. ``--port 6007``.

    Args:
        argv: Arguments after ``view``.

    Returns:
        TensorBoard's exit code.
    """
    parser = argparse.ArgumentParser(
        prog="tensorboard-book view",
        description="Show only some runs in TensorBoard. Other options "
        "are passed on to TensorBoard.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--logdir", nargs="+", required=True, help="run folders to show"
    )
    args, tensorboard_args = parser.parse_known_args(argv)
    missing = [p for p in args.logdir if not Path(p).is_dir()]
    if missing:
        error(f"{', '.join(missing)}: not a folder.")
        return 2
    return tbview.view(args.logdir, tensorboard_args)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="tensorboard-book",
        description="Bookkeeping for folders of TensorBoard runs.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=version_text()
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, with_root=True):
        if with_root:
            p.add_argument("root", nargs="?", default=".", help="runs folder")
        p.add_argument(
            "--depth", type=int, default=1, help="folder depth of runs"
        )
        p.add_argument(
            "--db",
            help="database path (default: ROOT/.tensorboard-book/index.db)",
        )
        p.add_argument(
            "--workers",
            type=int,
            default=min(4, os.cpu_count() or 1),
            help="processes used to parse runs (default: up to 4)",
        )

    serve = sub.add_parser("serve", help="start the web app (default)")
    common(serve)
    serve.add_argument(
        "--host", default="127.0.0.1", help="address to listen on"
    )
    serve.add_argument(
        "--port",
        type=int,
        help="port to listen on (default: 8501, or the next free one)",
    )
    serve.add_argument(
        "--no-auth",
        action="store_true",
        help="skip the password even if TBOOK_PASSWORD_HASH is set (only "
        "with a local --host)",
    )
    serve.add_argument(
        "--no-browser",
        action="store_true",
        help="don't open the app in a browser tab",
    )
    serve.add_argument(
        "--tensorboard-host",
        default="127.0.0.1",
        help="address for TensorBoards opened from the app (they have no "
        "password, so keep it local unless your network is trusted)",
    )
    serve.set_defaults(func=cmd_serve)

    scan = sub.add_parser(
        "scan", help="index a folder without starting the app"
    )
    common(scan)
    scan.add_argument(
        "--force", action="store_true", help="re-parse every run"
    )
    scan.set_defaults(func=cmd_scan)

    # Handled before parsing (it passes unknown options to TensorBoard);
    # listed here so it shows up in --help.
    sub.add_parser(
        "view", help="show only some runs in TensorBoard (tensorboard-view)"
    )

    hp = sub.add_parser(
        "hash-password", help="create the password hash variable"
    )
    hp.set_defaults(func=cmd_hash_password)

    demo = sub.add_parser("demo", help="create a folder of fake runs")
    demo.add_argument("dest", nargs="?", default="tbook-demo")
    demo.set_defaults(func=cmd_demo)

    exp = sub.add_parser(
        "export-annotations", help="write annotations to JSON"
    )
    common(exp)
    exp.add_argument("-o", "--output", default="tbook_annotations_export.json")
    exp.set_defaults(func=cmd_export)

    imp = sub.add_parser(
        "import-annotations", help="merge annotations from JSON"
    )
    imp.add_argument("root")
    imp.add_argument("file")
    common(imp, with_root=False)
    imp.set_defaults(func=cmd_import)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``tensorboard-book`` command.

    Args:
        argv: Arguments without the program name (defaults to ``sys.argv``).

    Returns:
        Exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "view":
        return cmd_view(argv[1:])
    if not argv or (
        argv[0] not in SUBCOMMANDS
        and argv[0] not in ("-h", "--help", "-v", "--version")
    ):
        argv.insert(0, "serve")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
