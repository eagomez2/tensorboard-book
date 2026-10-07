# Metric directions and settings

## Metric directions

For each metric, the app needs to know whether higher or lower is better.
Names containing `loss`, `error`, `mse`, `mae`, `wer`, `fid`
and similar words default to lower is better, and everything else to higher is
better. In **Manage → Metric directions** each metric is *auto* (follows
that guess) until you pick *higher* or *lower*. **Reset all
to auto** undoes every choice. Arrows (↑/↓) in column headers show the current choice.

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

The **Auto** toggle in the sidebar rescans every 60 seconds and refreshes the
page when something changed.
