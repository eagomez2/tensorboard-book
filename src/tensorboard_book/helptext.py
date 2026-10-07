"""In-app documentation shown in the help dialog, one Markdown text per topic.

The keys of ``HELP`` are the topics listed in the dialog. View names match the
sidebar views so each view opens its own topic.
"""

HELP = {
    "Overview": """
`tensorboard-book` keeps a database of every TensorBoard run in a
folder, so you can compare them, annotate them and find their files.

**What is a run?** A folder one level below the root (or `--depth` levels)
that contains `events.out.tfevents.*` files anywhere inside it. Event files
in subfolders (like Keras' `train/` and `validation/`) are merged into the
run, and their tags are prefixed with the subfolder name.

**How it stays up to date.** When you start the app, the folder is scanned
in the terminal. After that, **Rescan** in the sidebar (or **Auto**, every
60 s) re-reads only the runs whose event or config files changed, so it's
quick even with hundreds of runs.

**What's read from each run:**
- every scalar (PyTorch `add_scalar` and TF2/Keras summaries), with its best,
  last, min and max values and the step where they happened.
- hyperparameters from the TensorBoard hparams plugin and from `config`,
  `hparams`, `args`, `params`, `cfg` or `opts` files (`.yaml`, `.yml` or
  `.json`).
- text summaries.
- when events were written, to measure run time.

**Your annotations** (tags, notes, stars, groups, properties) live in the
database and are copied to `annotations.json` after every edit. Both files
are in a hidden `.tensorboard-book` folder inside the runs folder, so they
move with it but don't show up among the runs or their files.

**Every view is laid out the same way**, top to bottom:
1. the title, with this help on the right.
2. summary cards with the key numbers.
3. the controls for what to show (**Show** always adds extra columns).
4. the content.
5. a toolbar with actions and downloads: what's selected on the left,
   actions in the middle, downloads last.

The sidebar filters (group, tags, status, search) appear in the views
they change: Experiments and Timeline. **Help** at the end of the menu
opens this guide from anywhere.

Nothing in your runs folder is ever changed or deleted.
""",
    "Experiments": """
The main table, with one row per run.

**Columns.** Pick them in the three boxes at the top:
- **Metrics**: the value per run is the *best* one (highest or lowest,
  depending on the metric's direction, shown by ↑ and ↓) or the *last* one,
  depending on the **Value** switch. The best value in each column is
  highlighted. Your choice is saved per group.
- **Hyperparameters**: by default, the ones that differ between the runs
  shown.
- **Show**: extra columns such as status, steps, compute time, start, tags
  (as colored pills), groups and notes.

**Filter.** Narrow the table with conditions, e.g.
`optimizer.lr < 1e-3 and val/acc > 0.9`. See the *Filters* topic.

**Actions.** Tick rows and use the toolbar under the table:
- **Plot curves** opens the selected runs in Curves.
- **Compare** (two or more runs) opens them in Compare, even if they aren't
  in a group.
- **Details** (one run) opens it in Run details.
- **Open in TensorBoard** starts TensorBoard on the selected runs (see the
  *TensorBoard* topic).
- **Group** adds them to a group (or a new one) or removes them from the
  current one.
- **Tag** adds or removes tags.
- **Property** sets or clears a custom property on all selected runs.
- **Star** and **Archive**: archived runs are hidden unless you turn on
  *Show archived* in the sidebar. Their files are untouched.
- **Export CSV** saves the table as shown.

**The numbers on top.** The number of runs shown, how many are active (see
the ⓘ next to *Active now*), total compute time, and disk usage.
""",
    "Compare": """
Compares the runs of one group (an ablation study) fairly.

**What to compare.** A group, **Selected runs** (the runs you ticked before
pressing **Compare** in Experiments), or **Runs shown in Experiments** (the
runs that pass the sidebar filters). Use **Show** to add tags, notes,
status, groups or steps to the table, so you can check that you're
comparing the runs you meant to. When seeds are aggregated, a **runs**
column lists the runs behind each row.

**Selection metric.** For each run, the step where this metric is best is
chosen (the "checkpoint"). Every metric in **Metrics at that step** is shown
*at that same step*, using the logged value closest to it. That's how test
scores should be reported: picking the best test value on its own would be
cherry-picking.

**Baseline.** Pick a run (or, with seeds aggregated, a config) to show
each number's difference to it in brackets, e.g. `0.9414  (+0.0147)`.
Green means better and orange means worse, following the metric's
direction. The differences are also in the exports. With *None*, the
default, the table shows no differences.

**Aggregate seeds.** Collapses runs that differ only by their seed into one
row with mean ± standard deviation and the number of runs (`n`). Two ways to
decide that runs share a config:
- their names match after removing the seed pattern (`_seed1`, `-seed-2`,
  and so on). The pattern is a regular expression that can be edited.
- all hyperparameters match except the ones named `seed`.

**Below the table:**
- a dot plot of the selection metric (with error bars when aggregated).
- *What differs between these runs*: the hyperparameters that vary, with
  cells highlighted where they differ from the best run.
- *Which settings lead to the best result*: pick a **Target metric** and
  the **Settings** (hyperparameters and properties) to see, like
  TensorBoard's HParams dashboard, a parallel coordinates plot and the
  target against each setting, with the best run ringed. The parallel
  coordinates plot has one line per run, colored by the target. Drag along
  an axis to keep only the runs in that range, or drag an axis name to
  reorder the axes. Wide-ranging numbers such as learning rates use a log
  scale, and missing values are shown as a dash.
- exports as CSV, Markdown or LaTeX (with the best value in bold).

The cards on top show how many runs and configs are compared and which one
is best. The selection metric is saved with the group.

**Open in TensorBoard** (in the toolbar) starts TensorBoard on exactly the
compared runs. See the *TensorBoard* topic.

The link in the address bar opens this exact comparison, so it can be
shared or bookmarked. It keeps the group (or the runs, when they aren't a
group), the selection metric, the metrics at that step, the **Show**
columns, the seed options and the baseline. Opening it asks for the
password first if one is set.
""",
    "Curves": """
Overlays training curves of up to eight runs.

- **Runs** and **Metrics**: one chart per metric, one line per run. Each run
  keeps its color while you add or remove others.
- **Smoothing**: the same debiased exponential moving average as
  TensorBoard. The faint line is the raw data.
- **X axis**: steps, hours since the run started (to compare speed), or wall
  time.
- **Log y**: logarithmic y axis.

Long curves are thinned to at most 5000 points for plotting, always keeping
the first, last, lowest and highest points. Values in tables are always
computed from every point.
""",
    "Timeline": """
Shows when runs were running.

- **Bars**: one per run, from its first to its last logged event. Hover for
  the span, the compute time and the status. Click a bar to open the run.
  The dotted line is now.
- **Color by**: group, status or first tag.
- **Runs at once**: how many runs were running at each moment. Useful to
  spot idle or overloaded machines.
- **Compute hours per day**: a calendar of how many hours of runs happened
  each day. Pauses (see below) are not counted.

The cards on top count the runs in the range, their compute time, the
busiest day, and how many are running now.

**Span vs compute time.** A silence longer than the *pause* setting (6 h by
default) counts as a pause, e.g. a job that crashed and was resumed the next
day. *Span* is first to last event, and *compute* leaves out pauses. The
**Timing** tab of a run lists its segments.

Times come from the clock of the machine that ran each job and are shown in
your time zone. The start is the first logged event, so setup time before it
isn't included.
""",
    "Run details": """
Everything about one run.

**Top.** Status, groups, tags and flags, then compute time, wall span,
throughput (steps per second of compute), last step, time since the last
event, and disk size.

**TensorBoard** next to the badges opens this run in TensorBoard.

**Tags, notes and properties.** Press **Edit tags, notes and properties**
next to the badges, change tags, groups, star, archive flag, notes and
custom properties in the form, and press **Save**. Property values also
show as purple badges.

**Tabs:**
- **Artifacts**: every file in the run folder except event files. Select a
  file to preview it (images, text and config files, CSV/TSV tables, audio,
  video) and download it. Select several to download a zip. For text
  files, pick another run to see a diff of the same file, e.g.
  `config.yaml`.
- **Metrics**: every scalar with best, best step, last, min, max, number of
  points and NaN/inf count.
- **Hyperparameters**: the flattened config and where each value came from.
- **Text**: text summaries (e.g. from `add_text`).
- **Timing**: activity segments and the pauses between them.

The command to open this run in TensorBoard is at the bottom of the Metrics
tab.
""",
    "Manage": """
Housekeeping.

- **Groups**: create, rename, describe, change members, delete. Deleting a
  group keeps its runs.
- **Tags**: rename (renaming to an existing tag merges them) or delete.
- **Properties**: define custom properties (see the *Properties* topic),
  rename them, describe them or delete them.
- **Metric directions**: whether higher or lower is better for each
  metric. Names with *loss*, *error*, *mse* and similar words
  default to lower. *auto* follows that guess. Pick
  *higher* or *lower* to fix a metric, and **Reset all to auto** to undo
  every choice.
- **Disk usage**: per run and per group.
- **Settings**: the active threshold, the pause length, the seed pattern,
  the stored points per curve, the zip size limit, and the TensorBoard
  limit and extra arguments. Changing the pause or the points re-parses all
  runs.
- **Backup**: download or import the annotations JSON, and forget runs whose
  folders were deleted.

Read-only visitors only see Disk usage and Backup.
""",
    "Filters": """
The **Filter** box in Experiments takes conditions joined by `and`:

```
optimizer.lr < 1e-3 and val/acc > 0.9 and model ~ resnet
```

- **Keys**: hyperparameters (dotted names, as in the table), metric tags
  (using the Best/Last value shown), custom properties (by name, or
  `prop:name` if a hyperparameter has the same name), `run`, `status` or
  `steps`. Wrap keys with spaces in backticks.
- **Operators**: `==` (or `=`), `!=`, `<`, `<=`, `>`, `>=`, and `~` for a
  case-insensitive regular expression, e.g. `run ~ ^aug_`.
- Numbers compare as numbers, everything else as text. A misspelled key
  gets a "did you mean" suggestion.

The sidebar filters appear in Experiments and Timeline, the views they
change, and keep their values while you visit other views:
- **Search** looks for text in names, tags, notes and config values.
- **Group**, **Tags**, **Status**, **Starred only** and **Show archived**
  narrow down the runs.

Compare can use the same runs through **Runs shown in Experiments**.
Curves and Run details list every run that isn't archived.
""",
    "Status and time": """
**Active or stopped.** TensorBoard logs don't record that a run finished,
so the status says what can be known:
- **Active**: an event was written recently. The default is within the
  last 5 hours, so a run whose epochs take hours still counts as active.
  Change it in Manage → Settings.
- **Stopped**: nothing written for longer than that. The run may have
  finished, crashed, or been killed.
- **Missing**: the folder is gone. Its annotations are kept.
- Runs that logged NaN or inf values get a red *NaN/inf* badge.

**Run time:**
- **Span**: time from the first to the last event.
- **Compute**: span minus pauses. A pause is a silence longer than 6 h by
  default. Set the pause above your longest time between two logged values
  (e.g. one epoch), or it would cut normal training into pieces.
- **Throughput**: steps per second of compute.
""",
    "Properties": """
Properties are values you add to runs by hand, for things the logs don't
say or that you decide afterwards: a **number** (GPU hours, a human rating,
a cost) or a **category** (dataset, hardware, who ran it, a verdict).

1. Define a property once in **Manage → Properties**: a name, the type and
   an optional description.
2. Give runs a value in **Run details → Edit tags, notes and properties**,
   or for many runs at once in **Experiments**: select rows, then
   **Property** in the toolbar. For a category you can pick an existing
   value or type a new one.
3. Show them as columns with **Show** in Experiments and Compare. They sit
   next to the metrics, and categories show as colored pills. When Compare
   aggregates seeds, numbers are averaged and categories are listed.

Properties also work in the **Filter** box (`dataset == cifar-100`,
`rating >= 4`) and in the sidebar search, and they are saved in the
annotations backup with everything else.
""",
    "TensorBoard": """
**Open in TensorBoard** starts TensorBoard on exactly the runs you pick: the
selected rows in Experiments, the runs being compared in Compare, or the
run shown in Run details (the **TensorBoard** button). Use it to look at
images, histograms and anything else `tensorboard-book` doesn't show.

- The run folders are linked (not copied) into a temporary folder, so
  TensorBoard lists them under the same names.
- It starts in the background on the first free port from 6006. Press
  **Open in new tab** once it's ready. Pressing the button again for the
  same runs reuses the running TensorBoard.
- Running TensorBoards are listed below the toolbar with a **Stop** button.
  All of them stop when `tensorboard-book` exits. At most 5 run at once by
  default (Manage → Settings, which also takes extra TensorBoard
  arguments).
- TensorBoard has no password, so it only listens on this machine
  (127.0.0.1). If you reach `tensorboard-book` through an SSH tunnel, forward
  the TensorBoard port too, e.g. `ssh -L 6006:localhost:6006 server`.

From a terminal, `tensorboard-book view --logdir RUN [RUN ...]` does the
same in the foreground. Any other options are passed to TensorBoard.
""",
    "Password and access": """
- On `127.0.0.1` the password is optional: the app asks for one only when
  `TBOOK_PASSWORD_HASH` is set, and `--no-auth` skips it anyway. Any other
  `--host` needs the hash.
- The password is checked against a salted hash in `TBOOK_PASSWORD_HASH`,
  created with `tensorboard-book hash-password`. A second hash in
  `TBOOK_VIEWER_PASSWORD_HASH` gives read-only access: browsing and
  downloading, but no edits.
- Files can only be downloaded from a logged-in session. Links to files
  outside the runs folder (e.g. symlinks) are refused.
- To share the app beyond your machine, use an SSH tunnel or a reverse proxy
  with HTTPS. See *Password and deployment* in the documentation.
""",
}
