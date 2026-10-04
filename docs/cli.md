# Command line

```text
tensorboard-book [ROOT] [--host 127.0.0.1] [--port 8501] [--depth 1]
                 [--db PATH] [--workers N] [--no-auth] [--no-browser]
                 [--tensorboard-host 127.0.0.1]          scan, then start the app
tensorboard-book view --logdir RUN [RUN ...] [TB args]   TensorBoard on some runs
tensorboard-book scan ROOT [--depth N] [--force]         index without the app (cron)
tensorboard-book hash-password                           create the password hash
tensorboard-book demo [DEST]                             create demo runs
tensorboard-book export-annotations ROOT [-o FILE]       write annotations to JSON
tensorboard-book import-annotations ROOT FILE            merge annotations from JSON
tensorboard-book -v, --version                           print the version
```

Streamlit prints the address to open. Usage statistics are off, and the app
doesn't watch its source files for changes (a `.streamlit/config.toml` in the
repository does the same for `streamlit run` during development).

Starting the app first scans the folder in the terminal, using up to four
processes (`--workers`), so the page opens with everything indexed. Rescans
from the app (the **Rescan** button and **Auto**) run inside the app in a
single process and only re-read runs that changed. As a rough guide, 300 runs
with 6.6 million scalar values (290 MB) take about 35 s with one worker and
18 s with four. An unchanged folder rescans in under a second.

## Options of each command

The output of `--help` for each command, as installed.

### `tensorboard-book`

`-v` or `--version` prints the version and the years of the project, for
example:

```console
$ tensorboard-book --version
tensorboard-book version 0.7.0 2026
```

From 2027 on it shows a range, such as `2026 - 2027`.

<!-- cli: -->

### `tensorboard-book serve`

The default command: `tensorboard-book ROOT` is the same as
`tensorboard-book serve ROOT`.

<!-- cli: serve -->

### `tensorboard-book view`

<!-- cli: view -->

### `tensorboard-book scan`

<!-- cli: scan -->

### `tensorboard-book hash-password`

<!-- cli: hash-password -->

### `tensorboard-book demo`

<!-- cli: demo -->

### `tensorboard-book export-annotations`

<!-- cli: export-annotations -->

### `tensorboard-book import-annotations`

<!-- cli: import-annotations -->
