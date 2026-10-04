# Status and run time

<!-- help: Status and time -->

## In more detail

TensorBoard logs have no "finished" marker, so these are estimates from the
event timestamps.

- "Best" is the best logged value, not a smoothed one.
- Other metrics in **Compare** use the logged step closest to the best step.
  If they were logged at different intervals, the step shown may differ
  slightly.
- Plotted curves are downsampled above the point limit (the reported numbers
  are not).
- Only scalars and text are shown. Images, histograms and other plugins
  are not shown.
