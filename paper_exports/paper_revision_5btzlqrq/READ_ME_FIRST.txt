PAPER HANDOFF PACKAGE — new revision experiments only
Figures are generated from full-precision CSVs, not rounded console output.
FORECAST FIGURES: bars are means; error bars are sample SD across training seeds,
not confidence intervals. Weekly naive is deterministic. Seed counts are labeled.
Caltech original-model five-seed results combine original and added runs; retain
the original three-seed table separately and disclose execution environments.
UrbanEV volume is estimated energy. Caltech hourly energy and UrbanEV regional
volume have different scales and should not be pooled into one error metric.
ALIGNMENT FIGURE: positive = lower MAE for aligned graph versus mean shuffled
graph within each seed. Panels have different y-axis scales to expose small
UrbanEV effects; inspect axis labels and disclose this in the paper caption.
RSPS SCATTER: descriptive only; points share observations and topology. RSPS
uses validation residual MSE; neural MAE uses test hourly forecasts. No fit line
or significance claim is included. This is a retrospective extension.
Captions belong in LaTeX; full captions have not been embedded in graphics.

This package contains result tables, settings, small evidence files, source
copies, and new figures. It excludes raw/input data, large prediction arrays,
model checkpoints, and the original Washington figures. It is NOT a full backup.
Keep a separate copy of the complete repository and revision_runs/revision_inputs.
The manuscript has not yet been edited by this export script.
