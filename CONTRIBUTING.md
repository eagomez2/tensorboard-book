# Contributing

## Development setup

```bash
git clone https://github.com/eagomez2/tensorboard-book
cd tensorboard-book
uv sync                       # includes the dev group (pytest, ruff)
uv run pytest                 # run the tests
uv run ruff check src tests   # PEP 8 (79 columns), Google-style docstrings
uv run ruff format src tests
uv build                      # sdist and wheel with the uv build backend
```

Tables are [polars](https://pola.rs) DataFrames throughout. The one place
pandas appears is the last step before a table with highlighted cells is
drawn, because Streamlit only supports per-cell styling through pandas'
Styler; pandas comes with Streamlit and isn't a direct dependency.

The code is flat, with one module per concern and no framework layers:

| Module | Contents |
|---|---|
| `scanner.py` | Finds runs, reads event files (a fast TFRecord reader), parses configs, and syncs the database. |
| `db.py` | SQLite schema, queries, annotation edits, and JSON backup and restore. |
| `analysis.py` | Pure helpers: metric tables, filter language, smoothing, seed keys, timing, exports. |
| `tbview.py` | Opens a selection of runs in TensorBoard (symlinks, ports, background instances). |
| `auth.py` | Password hashing and the login gate. |
| `helptext.py` | The in-app documentation, one Markdown text per topic. |
| `app.py` | The Streamlit UI, one function per view. |
| `brand.py` | Draws the logo; `python -m tensorboard_book.brand` rewrites it. |
| `assets/` | The logo, bundled with the package. |
| `static/fonts/` | IBM Plex Sans and Mono and Newsreader (SIL Open Font License), served by Streamlit so the app works offline. |
| `cli.py` | The `tensorboard-book` command. |
| `demo.py` | Demo data generator. |

The logo for the README lives in `assets/` at the top level.

## Documentation

The site is built with [MkDocs](https://www.mkdocs.org) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), from
`mkdocs.yml` and the `docs/` folder:

```bash
uv run --group docs mkdocs serve            # preview at http://127.0.0.1:8000
uv run --group docs mkdocs build --strict   # write the site to site/
```

The `docs` workflow in `.github/workflows/` publishes the site on GitHub
Pages at <https://eagomez2.github.io/tensorboard-book/> on every push to
`main`. For it to work, set **Settings → Pages → Source** to
**GitHub Actions** in the repository.

The in-app help and the command line help aren't copied into `docs/`, so
the site always says the same as the app:

- `<!-- help: Compare -->` is replaced by that topic of the in-app help, from
  `src/tensorboard_book/helptext.py`.
- `<!-- cli: serve -->` is replaced by the output of
  `tensorboard-book serve --help`.
- The logo and fonts are published from the package.

`docs/_theme/hooks.py` does all three. The colors and fonts in
`docs/stylesheets/extra.css` are the app's. Screenshots are in
`docs/images/`, one for each color scheme, taken from the demo data at a
1440 × 900 window.

The README is kept short on purpose: what the tool is, how to install and
start it, and a link to the documentation. Everything else belongs in
`docs/`.
