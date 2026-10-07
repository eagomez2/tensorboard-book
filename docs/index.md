# TensorBoard Book

`tensorboard-book` is a web application for organizing and comparing the
runs in a TensorBoard log folder. It indexes every run, reports the best
value of each metric, and stores tags, notes and groups alongside the runs.

![The Experiments view](images/experiments-light.webp#only-light){ .tb-shot }
![The Experiments view](images/experiments-dark.webp#only-dark){ .tb-shot }

## Features

- A table of all runs with their best metrics, hyperparameters and filters.
- Tags, notes, stars, groups and custom properties for each run.
- Comparison of a group at the step where a selection metric is best, with
  mean ± std over seeds and CSV, Markdown or LaTeX export.
- Overlaid training curves with TensorBoard's smoothing.
- A timeline of when runs were active, and compute hours per day.
- A file browser with previews, diffs and downloads for each run.
- TensorBoard on any selection of runs.

`tensorboard-book` reads the folder you pass to `tensorboard --logdir` and
never modifies it.

## Getting started

1. [Installation](install.md): install the command and start the app.
2. [Quick start](quickstart.md): a tour of the app on demo data.

## Documentation

- [Your runs folder](runs.md): what counts as a run, and where the app
  stores its data.
- [The views](views/index.md): each page of the app.
- Topics: [filters](topics/filters.md),
  [status and run time](topics/status-and-time.md),
  [properties](topics/properties.md),
  [opening TensorBoard](topics/tensorboard.md),
  [metric directions and settings](topics/settings.md).
- [Password and deployment](deployment.md): sharing the app safely.
- [Command line](cli.md): all commands and options.

## Disclaimer

`tensorboard-book` is an independent project. It is not affiliated with or
endorsed by Google or the TensorFlow team.
