"""In-app documentation shown on the Help page, one Markdown text per topic.

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

**Your annotations** (tags, notes, stars, collections, fields) live in the
database and are copied to `annotations.json` after every edit. Both files
are in a hidden `.tensorboard-book` folder inside the runs folder, so they
move with it but don't show up among the runs or their files.

**Every view is laid out the same way**, top to bottom:
1. the title, with this help on the right.
2. summary cards with the key numbers.
3. the controls for what to show (**Show** always adds extra columns).
4. the content.
5. under a table, a row of buttons: how many rows are
   selected, the actions on them (greyed out until they apply), and the
   downloads on the right.

Every chart has a **Download figure** button: it saves the chart as a
plotly JSON file, which reports that take plotly figures show as an
interactive chart.

The sidebar filters (collection, tags, status, search) appear in the views
they change: Experiments and Timeline. **Help** at the end of the menu
opens this guide, every topic on one page.

Nothing in your runs folder is ever changed or deleted.
""",
    "Experiments": """
The main table, with one row per run.

**Columns.** Pick them in the three boxes at the top:
- **Metrics**: the value per run is the *best* one (highest or lowest,
  depending on the metric's direction, shown by ↑ and ↓) or the *last* one,
  depending on the **Value** switch. The best value in each column is
  highlighted. Your choice is saved per collection.
- **Hyperparameters**: by default, the ones that differ between the runs
  shown.
- **Show**: extra columns such as status, steps, compute time, start, tags
  (as colored pills), collections, notes and your fields. Fields are listed
  with their group, as `Model · params (M)`; see the *Fields* topic.

**Filter.** Narrow the table with conditions, e.g.
`optimizer.lr < 1e-3 and val/acc > 0.9`. See the *Filters* topic.

**Actions.** Tick rows and use the buttons under the table. They are
greyed out until they apply:
- **Open in**: **Run details** (one run), **Curves**, or **TensorBoard**,
  which starts TensorBoard on the selected runs (see the *TensorBoard*
  topic).
- **Compare** (two or more runs) opens them in Compare, even if they aren't
  in a collection.
- **Add to collection** adds them to a collection (or a new one) or
  removes them from the current one.
- **Field** sets or clears a field on all selected runs. Pick **Tags** to
  add or remove tags.
- **Star** and **Archive**: archived runs are hidden unless you turn on
  *Show archived* in the sidebar. Their files are untouched.
- **Export CSV** saves the table as shown.

**The numbers on top.** The number of runs shown, how many are active (see
the ⓘ next to *Active now*), total compute time, and disk usage.
""",
    "Compare": """
Compares the runs of one collection (an ablation study) fairly.

**What to compare.** A collection, **Selected runs** (the runs you ticked
before pressing **Compare** in Experiments), or **Runs shown in
Experiments** (the runs that pass the sidebar filters). Use **Show** to add
tags, notes, status, collections, steps or fields to the table, so you can
check that you're comparing the runs you meant to. When seeds are
aggregated, a **runs** column lists the runs behind each row, number
fields are averaged and choice fields list their values.

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
  the **Settings** (hyperparameters and number fields) to see, like
  TensorBoard's HParams dashboard, a parallel coordinates plot and the
  target against each setting, with the best run ringed. The parallel
  coordinates plot has one line per run, colored by the target. Drag along
  an axis to keep only the runs in that range, or drag an axis name to
  reorder the axes. Wide-ranging numbers such as learning rates use a log
  scale, and missing values are shown as a dash.
- exports as CSV, Markdown or LaTeX (with the best value in bold).

The cards on top show how many runs and configs are compared and which one
is best. The selection metric is saved with the collection.

**Open in TensorBoard** (under the table) starts TensorBoard on exactly the
compared runs. See the *TensorBoard* topic.

The link in the address bar opens this exact comparison, so it can be
shared or bookmarked. It keeps the collection (or the runs, when they
aren't one), the selection metric, the metrics at that step, the **Show**
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
- **Overlay metrics**: below the charts, draws several metrics of the chosen
  runs on one chart, to compare metrics on the same scale (for example
  `train/loss` and `val/loss`). It starts with metrics that share a name
  after the last `/`. **Color by** run or metric; the other one is told
  apart by the line style.

Long curves are thinned to at most 5000 points for plotting, always keeping
the first, last, lowest and highest points. Values in tables are always
computed from every point.
""",
    "Timeline": """
Shows when runs were running.

- **Bars**: one per run, from its first to its last logged event. Hover for
  the span, the compute time and the status. Click a bar to open the run.
  The dotted line is now.
- **Color by**: collection, status or first tag.
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

**Top.** Status, collections, tags and flags, then compute time, wall span,
throughput (steps per second of compute), last step, time since the last
event, and disk size.

**TensorBoard** next to the badges opens this run in TensorBoard.

**Edit** (next to **TensorBoard**) opens the editor in place of the
page: one card per field group, with Tags
under *General*, then the run's collections, star and archive flag, then
its notes. Each field has the input of its type: a number (typed in the
field's unit, and it can end in a prefix such as `25.6M`), a choice or
tags list (type a new value to add it as an option), yes/no, a date, a
link or text. **Add a field** defines a new one without leaving the
editor. Press **Save changes**, or **Cancel**.

**Tabs:**
- **Overview**: the run's fields in cards, one per field group, with
  units and colored options, then its collections. Empty fields are hidden
  unless you turn on **Show empty fields**.
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
    "Collections": """
Collections are named sets of runs, such as a sweep or an ablation study.
A run can be in several collections, and deleting a collection keeps its
runs.

- **New collection**: a name and an optional description. You can also
  select runs in Experiments and use **Add to collection** under the
  table.
- **Edit collection**: rename it, describe it, change its runs, or delete
  it.
- Each collection remembers its selection metric (from Compare) and its
  metric columns (from Experiments).

Pick a collection in the sidebar **Collection** filter to see only its
runs, or in Compare to compare them.
""",
    "Fields and settings": """
Everything about how this runs folder is described, in four tabs.

- **Fields**: the fields you add to runs (see the *Fields* topic). Edit
  the table and press **Save fields**: add a row to create a field, change
  a name to rename it (its values follow), change its group, type, unit,
  scale, decimals, direction or options, or untick **Visible** to keep it
  out of **Show** by default. Changing the type converts the values and
  stops with a message if one doesn't fit. Tick **Delete** to remove a
  field and its values when you save. **Rename values** changes a value
  on every run that has it: fix a typo, merge two spellings such as
  `resnet` and `ResNet`, reorder (which sets the colors) or delete the
  values of a choice or tags field, including the built-in `Tags`. The
  table only edits the list a field offers, and keeps the values runs
  still use. **Export
  fields** and **Import fields** copy the definitions to another runs
  folder; values are not included.
- **Metrics**: for each logged metric, its group (the tag up to the last
  `/` unless you change it), whether higher or lower is better, its unit,
  scale and decimals. Names with *loss*, *error*, *mse* and similar words
  default to lower; *auto* follows that guess. **Reset all to auto**
  forgets every direction you set.
- **Settings**: the active threshold, the pause length, the seed pattern,
  the stored points per curve, the zip size limit, and the TensorBoard
  limit and extra arguments. Changing the pause or the points re-parses all
  runs.
- **Project**: where the runs folder and database are, disk usage per run
  and per collection, and **Backup**: download or import the annotations
  JSON, and forget runs whose folders were deleted.

Read-only visitors see the fields and metrics but can't change them.
""",
    "Filters": """
The **Filter** box in Experiments takes conditions joined by `and`:

```
optimizer.lr < 1e-3 and val/acc > 0.9 and model ~ resnet
```

- **Keys**: hyperparameters (dotted names, as in the table), metric tags
  (using the Best/Last value shown), fields (by name, or `prop:name` if a
  hyperparameter has the same name), `run`, `status` or
  `steps`. Wrap keys with spaces in backticks.
- **Operators**: `==` (or `=`), `!=`, `<`, `<=`, `>`, `>=`, and `~` for a
  case-insensitive regular expression, e.g. `run ~ ^aug_`.
- Numbers compare as numbers, everything else as text. A misspelled key
  gets a "did you mean" suggestion.

The sidebar filters appear in Experiments and Timeline, the views they
change, and keep their values while you visit other views:
- **Search** looks for text in names, tags, notes and config values.
- **Collection**, **Tags**, **Status**, **Starred only** and **Show archived**
  narrow down the runs.

Compare can use the same runs through **Runs shown in Experiments**.
Curves and Run details list every run that isn't archived.
""",
    "Status and time": """
**Active or stopped.** TensorBoard logs don't record that a run finished,
so the status says what can be known:
- **Active**: an event was written recently. The default is within the
  last 5 hours, so a run whose epochs take hours still counts as active.
  Change it in Fields and settings → Settings.
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
    "Fields": """
Fields are values you add to runs by hand, for things the logs don't say
or that you decide afterwards: the dataset, the parameter count, the GPU,
a rating, whether someone reviewed the run, a link to the write-up.

**Each field has:**
- a **name** and an optional **group** (such as *Data*, *Model* or
  *Review*). Fields of a group stay together everywhere: in the Overview
  cards, in the Edit form and in **Show** (`Model · params`).
- a **type**: *number*, *text*, *choice* (one of a list), *tags* (several
  of a list), *yes/no*, *date* or *link*.
- for numbers, a **unit**, a **scale** prefix (k, M, G, m, µ, Ki, Mi, …)
  and **decimals** for display, and whether **higher or lower is better**.
  Values are stored as entered, so `25.6M` with scale *M* shows as
  `25.6 M` and still sorts and filters as 25,600,000. A number that ranks
  gets ↑ or ↓ in its header and its best value highlighted, like a metric.
- for choice and tags, an ordered list of **options**. An option's place
  in the list gives its color, so a value keeps its color everywhere.
  Typing a value that isn't an option yet adds it at the end.
- a **description**, shown as help next to its input.

**Using them:**
1. Define fields in **Fields and settings → Fields**.
2. Give runs values in **Run details → Edit**, or for many runs at once in
   **Experiments**: select rows, then **Field** in the toolbar.
3. Show them as columns with **Show** in Experiments and Compare, filter
   on them (`Dataset == cifar-100`, `Params < 20M`, `Rating >= 4`) and
   find them with the sidebar search.

The run tags are the built-in `Tags` field: its options and colors are
managed the same way, and it can't be renamed or deleted. Fields are saved
in the annotations backup with everything else.
""",
    "TensorBoard": """
**TensorBoard** starts TensorBoard on exactly the runs you pick: the
selected rows in Experiments (under **Open in**), the runs being compared
in Compare (**Open in TensorBoard**), or the
run shown in Run details (the **TensorBoard** button). Use it to look at
images, histograms and anything else `tensorboard-book` doesn't show.

- The run folders are linked (not copied) into a temporary folder, so
  TensorBoard lists them under the same names.
- It starts in the background on the first free port from 6006. Press
  **Open in new tab** once it's ready. Pressing the button again for the
  same runs reuses the running TensorBoard.
- Running TensorBoards are listed below the toolbar with a **Stop** button.
  All of them stop when `tensorboard-book` exits. At most 5 run at once by
  default (Fields and settings → Settings, which also takes extra TensorBoard
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
