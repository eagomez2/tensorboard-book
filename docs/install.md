# Installation

## Requirements

- Python 3.10 or newer.
- [uv](https://docs.astral.sh/uv/) (recommended) or pip.

## Install

Install the `tensorboard-book` command from a copy of the repository:

```bash
git clone https://github.com/eagomez2/tensorboard-book
cd tensorboard-book
uv tool install .       # or: pip install .
```

Check the installation:

```bash
tensorboard-book --version
```

To work on the code instead, see
[CONTRIBUTING.md](https://github.com/eagomez2/tensorboard-book/blob/main/CONTRIBUTING.md).

## Start the app

Pass the folder that contains your runs, the same folder you would pass to
`tensorboard --logdir`:

```bash
tensorboard-book /path/to/runs
```

The command first indexes the folder, then prints the address of the app:

```text
  tensorboard-book 0.7.0

  Runs folder: /path/to/runs
  Database: /path/to/runs/.tensorboard-book/index.db
  Runs: 14 (14 new, 14 parsed)

  You can now view your Streamlit app in your browser.

  URL: http://127.0.0.1:8501
```

- **Browser.** The app opens in a new browser tab. `--no-browser` turns
  this off. On a machine without a display, such as over SSH, no tab is
  opened.
- **Port.** The app listens on port 8501, or on the next free port if 8501
  is in use. With `--port`, that exact port is used, and the app stops if it
  is busy.
- **Address.** The app listens on `127.0.0.1` by default, so only the same
  computer can reach it. Any other `--host` requires a password.

## Set a password

Without a password hash, the app does not ask for a password. This is only
allowed on `127.0.0.1`. To require a password:

1. Create a password hash. The password itself is never stored.

    ```bash
    tensorboard-book hash-password
    # Password: ********
    # Repeat:   ********
    # TBOOK_PASSWORD_HASH=5c1f…:9ae2…
    ```

2. Set it in the environment and start the app:

    ```bash
    export TBOOK_PASSWORD_HASH='5c1f…:9ae2…'
    tensorboard-book /path/to/runs
    ```

When `TBOOK_PASSWORD_HASH` is set, the app asks for the password.
`--no-auth` skips it for one launch, on `127.0.0.1` only. The app does not
start if the hash is malformed, or if it is missing while `--host` is not
local.

`TBOOK_VIEWER_PASSWORD_HASH`, created the same way, sets a second,
read-only password. Viewers can browse and download, but not edit.

To reach the app from another computer, see
[Password and deployment](deployment.md).

## Next steps

- [Quick start](quickstart.md): a tour of the app on demo data.
- [Your runs folder](runs.md): what counts as a run, and where
  `tensorboard-book` stores its data.
- [Command line](cli.md): all commands and options.
