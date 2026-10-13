# The views

Pick a view from the menu in the sidebar. The current one is highlighted.
The menu has three sections:

- **Explore**: Experiments, Run details, Compare, Curves and Timeline.
- **Organize**: Collections.
- **Project**: Fields and settings, and Help.

Every view is laid out the same way, top to bottom:

1. The title, with a **How this page works** button on the right.
2. Summary cards with the key numbers.
3. Controls for what to show. **Show** always adds extra columns.
4. The content.
5. Under a table, a row of buttons: how many rows are selected, the
   actions on them (greyed out until they apply), and the downloads on
   the right. Charts have a **Download figure** button that saves them as
   plotly JSON.

**Help**, the last item of the menu, is a page with every guide, one topic
at a time: the per-view guides, plus filters, status and run time,
fields, TensorBoard and password access.
The bottom of the sidebar shows how many runs there are and how many are
active, when they were last scanned, **Rescan** and **Auto**, and the
version you're running. The app follows your system's light or dark
setting. To pick one, open the ⋮ menu in the sidebar's bottom-left corner
and choose **Settings → Theme**.

The sidebar **filters** (collection, tags, status, search) only appear in
the views they change, Experiments and Timeline, and keep their values
while you visit other views. The **★ Starred only** and **Show archived**
pills sit in those views, under the filters.

On/off options, such as **Aggregate seeds** in Compare or **Log y** in
Curves, are pills: click one to turn it on, and again to turn it off. Compare can use the same runs
through **Runs shown in Experiments**. Curves and Run details list every
run that isn't archived.

The pages in this section are the same guides the app shows under
**How this page works**:

| View | What it's for |
|---|---|
| [Experiments](experiments.md) | The main table: every run with its best scalars, filters, tags, collections and fields. |
| [Run details](run-details.md) | Everything about one run: its fields, files, scalars and timing. |
| [Compare](compare.md) | A fair comparison of a collection, with seeds aggregated and exports for papers. |
| [Curves](curves.md) | Training curves of up to eight runs, overlaid and smoothed. |
| [Timeline](timeline.md) | When runs were active, how many ran at once, compute hours per day. |
| [Collections](collections.md) | Named sets of runs, such as a sweep or an ablation study. |
| [Fields and settings](fields-and-settings.md) | Fields, scalar display, settings, disk usage and backups. |
