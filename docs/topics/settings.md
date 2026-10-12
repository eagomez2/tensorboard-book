# Scalars and settings

Both are in **Fields and settings**, in the **Scalars** and **Settings**
tabs.

## Scalars

Each logged scalar has:

- a **group**: the tag up to the last `/` (`val` for `val/acc`), unless you
  change it.
- a **direction**: whether higher or lower is better. Names containing
  `loss`, `error`, `mse`, `mae`, `wer`, `fid` and similar words default to
  lower is better, and everything else to higher is better. Each scalar is
  *auto* (follows that guess) until you pick *higher* or *lower*. **Reset
  all to auto** forgets every direction you set. Arrows (↑/↓) in column
  headers show the current choice.
- a **unit**, a **scale** prefix and **decimals**, for display.

## Settings

| Setting | Default | Notes |
|---|---|---|
| Active if written to within | 300 min (5 h) | Status threshold. |
| Pause that splits a run into segments | 360 min (6 h) | Changing it re-parses every run. |
| Seed pattern | `[_-]?seed[_-]?\d+` | Used to aggregate seeds by run name. |
| Max stored points per curve | 5000 | Longer curves are downsampled for plotting, always keeping the first, last, min and max points. Best, min, max and last values are always computed from every point. Changing it re-parses every run. |
| Max zip download | 1024 MB | |
| Max TensorBoards at once | 5 | Each instance is a separate process. |
| Extra TensorBoard arguments | (empty) | Added to every launch, e.g. `--samples_per_plugin images=100`. |
| Config file names | `config.*, hparams.*, args.*, params.*, cfg.*, opts.*` | Patterns of the `.yaml`, `.yml`, `.json` or `.toml` files read as hyperparameters. See [Config files](../runs.md#config-files). |

The **Auto** toggle in the sidebar rescans every 60 seconds and refreshes the
page when something changed.
