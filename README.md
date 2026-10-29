# Comparative Assessment of Physics-Neural Hybrids against PINNs and Neural ODEs for Trajectory Reconstruction and Parameter Recovery

Training code, experiment configurations and final numerical results for the
gene-regulatory and eco-epidemic benchmarks in the manuscript.

## Contents

| Directory | Contents |
| --- | --- |
| [native](native) | Five models, two systems, four noise levels and five seeds; 200 runs |
| [ablations](ablations) | Component ablations and controls; 59 records |
| [matched](matched) | Completeness and guidance controls; 216 runs |
| [results](results) | Final metrics, parameter estimates, reported comparisons, sensitivity results and computational accounting |

[Results guide](results/README.md) maps the released files to the manuscript
sections, figures and tables. The release contains 475 final experimental
records. These are not 475 independent replications of the same comparison.

## Run the experiments

Use Python 3.10 or newer and install `requirements.txt`.

```sh
pip install -r requirements.txt
python native/train.py --all --dry-run
python native/train.py --config native/configs/ma_node_gene_noise00.yaml --seed 42 --device cuda
python ablations/train.py --all --dry-run
python matched/train.py --seed 1031 --dry-run
```

Native seeds are 42, 123, 2024, 3407 and 7777. Matched-control seeds are 1031,
70595 and 250333. Each suite retains its own implementation and settings.
Training writes new outputs to the selected `--output` directory.

## Release scope

The release includes final per-run metrics and aggregate numerical evidence.
The loss-curve CSV contains five-seed summaries used for Figure 5, not individual
training logs. Old logs, checkpoints, development attempts and manuscript
working files are not included.

Four PG-NODE continuation configurations need initial weights that are not
included; they are skipped by the ablation `--all` command unless
`--include-continuations` is specified. Twelve eco full-RHS forward variants for
matched seed 250333 involve resumed training with incomplete earlier-stage
source provenance. Their archived final results are included, but a new run of
the supplied implementation is not claimed to reconstruct those runs exactly.
MC and MA labels in the matched suite are not independent seed replications.

- [External initial weights](EXTERNAL_WEIGHTS.txt)
- [Training and reproduction notes](TRAINING_SUPPLEMENT.txt)
- [Code file guide](CODE_OVERVIEW.txt)

No complete-budget retraining is claimed by this release. The repository does
not include the manuscript source or a new license grant.
