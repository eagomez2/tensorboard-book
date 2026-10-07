# Your runs folder

## What counts as a run

A run is a folder, one level below the root, that contains
`events.out.tfevents.*` files anywhere inside it:

```
runs/                     <- tensorboard-book runs
├── resnet_lr3e-4/        <- a run
│   ├── events.out.tfevents.…
│   ├── config.yaml       <- becomes hyperparameters
│   ├── checkpoints/…     <- artifacts
│   └── samples/…
└── keras_baseline/       <- also one run
    ├── train/events…     <- tags become train/epoch_loss, …
    └── validation/events…
```

- Event files in subfolders are merged into the run, with the subfolder as a
  tag prefix (Keras' `train/` and `validation/`). PyTorch's `add_hparams`
  timestamp subfolders are merged without a prefix.
- If your runs are nested deeper, use `--depth`. For example, with
  `runs/project/exp1/events…`, use `tensorboard-book runs --depth 2`.
- Both PyTorch-style scalars (`simple_value`) and TF2 tensor scalars are read.
- Hyperparameters come from the TensorBoard hparams plugin and from
  `config`, `hparams`, `args`, `params`, `cfg` or `opts` files (`.yaml`,
  `.yml` or `.json`) at the top of the run, one folder down, or in
  `.hydra/`. Lightning's `hparams.yaml` with `!!python/object` tags is read
  safely, and nothing is ever instantiated.
- Text summaries are shown on the run page. Images, histograms and other
  plugins are counted but not rendered: open TensorBoard for those. The run
  page shows the exact `tensorboard --logdir …` command.

## Renamed, moved, copied and deleted runs

Runs are identified by a fingerprint of their first event file, not by their
path, so renaming or moving a run folder keeps its tags, notes and groups. A
copied folder becomes a new run. A deleted folder is marked **Missing** and
keeps its annotations until you choose **Manage → Backup → Forget missing
runs**. `tensorboard-book` never modifies or deletes anything in your runs
folder, apart from its own two files.

## Where data is stored

| File | What | Safe to delete? |
|---|---|---|
| `ROOT/.tensorboard-book/index.db` | SQLite index (WAL mode). | Yes. It is rebuilt from the event files, and annotations are restored from the backup below. |
| `ROOT/.tensorboard-book/annotations.json` | Your tags, notes, stars, groups, properties, metric directions and settings. It is rewritten after every edit. | No. Commit it to git or back it up. |

The folder is hidden, so it doesn't clutter the runs folder, isn't taken
for a run and doesn't appear in the artifact browser, but it is copied or
moved along with the runs folder. Files from versions before 0.6
(`ROOT/.tbook.db`, `ROOT/tbook_annotations.json`) are moved into it on the
next start.

If the runs folder isn't writable, both files go to
`~/.cache/tensorboard-book/<folder name>-<hash>/`. You can also choose the location with
`--db PATH`, and the annotations file goes next to it. On first start with an
empty database, annotations are restored from the JSON automatically. You can
also export and import them from **Manage → Backup** or with the
`export-annotations` and `import-annotations` commands.
