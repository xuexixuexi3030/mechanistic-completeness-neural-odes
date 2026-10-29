# Numerical results

Values retain the archived evaluation definitions and numerical precision.
CSV files use decimal fractions for noise and relative errors unless a column
explicitly states otherwise. Empty cells mean unavailable or inapplicable,
not zero. JSON files retain additional final metrics; NPZ files contain
numerical arrays and can be opened with NumPy.

## Manuscript map

| Manuscript location | Files |
| --- | --- |
| Section 5.1; Figure 5 | `native/five_seed_loss_curves.csv`, `native/loss_endpoints.csv` |
| Sections 5.2.1-5.2.3; Figures 6-8; Tables 1-2 and S5-S7 | `native/run_metrics.csv`, `native/five_seed_summary.csv`, `native/parameter_estimates.csv`, `native/high_noise_parameter_errors.csv` |
| Section 5.2.4; Figures 9-10; Table 3 | `native/paired_comparisons.csv`, `native/paired_conditions.csv`, `native/paired_seed_blocks.csv` |
| Section 5.2.5; Table 4 and S1-S4 | `computational_cost.csv`, `native/training_budgets.csv`, `ablations/run_metrics.csv`, `matched/run_metrics.csv`; configurations in the training directories |
| Section 5.3; Figures 11-14 | `ablations/plotted_component_ratios.csv`, `ablations/run_metrics.csv` |
| Section 5.4; Figures 15-16; Tables S8-S9 | `sensitivity/summary.json`, `sensitivity/perturbations.json`, `sensitivity/fitted_diagnostics.json`, `sensitivity/numerical_checks.csv`, and the NPZ arrays |
| Section 5.5; Figures 17-18 | `matched/run_metrics.csv`, `matched/completeness_effects.csv`, `matched/pathway_comparisons.csv`, `matched/final_metrics.json` |

## Interpretation

- Native results contain 200 records. Shared parameter error averages `k_on`
  and `k_off` for gene, and `r`, `K` and `gamma` for eco. Neural ODE does not
  estimate physical parameters. `native/final_metrics.json` also retains
  numeric final diagnostics from the archived metrics.
- Paired comparisons contain 600 condition-level ratios and 75 seed-block
  ratios across 15 comparisons. Confidence intervals are pointwise 95% t
  intervals over five seed blocks. Exact two-sided sign-flip P values enumerate
  all 32 sign assignments; raw and Holm-adjusted values are both retained.
- Loss endpoints are the final common logged epoch for the five-seed curve.
  Per-run last logged losses need not be the selected checkpoint losses.
  Raw objectives differ between model families and are not a common accuracy
  metric. The curve file contains aggregate plotting data, not raw logs.
- The 59 ablation/control records include four continuations. Figures 11-13
  use the 51 rows in `plotted_component_ratios.csv` and their own within-family
  reference runs. Parameter errors are within-model quantities and may use a
  different parameter set from the native shared-parameter metric. Parameter
  inference is inapplicable when the physical field/skeleton is absent.
- Matched results contain 216 records. `run_configurations.json` gives each
  setting, `parameter_estimates.csv` contains the final parameters, and
  `final_predictions.csv` and `observations.csv` provide the saved numerical
  inputs for the reported comparisons. Observations are shared within each
  system/noise/seed block. Completeness and pathway ratios use double-precision
  errors recomputed from saved predictions; their last digits can differ
  slightly from archived single-precision metrics. Final gradient diagnostics
  and the 1% parameter-intervention response are separate quantities.
- Sensitivity arrays and summaries are the reported analysis outputs. The two
  observation CSV files are their reference inputs. Machine-specific source
  paths were replaced with these relative filenames; numerical values were
  preserved. Figure-assembly and sensitivity-analysis scripts are not part of
  this training-code release.
- Computational cost sums recorded timers for the 200 native runs, 59 ablation/
  control records and 216 matched runs. The sum is 1815.59 recorded hours.
  Native/ablation timers include trainer construction and training; matched
  timers cover training and checkpoint writes. Post-training evaluation,
  failed attempts, development and trial-and-error costs are excluded. These
  are not hardware-normalized GPU-hours or elapsed project time. Continuations
  are accounted for using the timer conventions described in the manuscript.

The final metrics are evidence from the archived experiments, not results of a
new complete-budget rerun. See the root training notes for reproduction scope.
