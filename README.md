# Sia_STP_Net

## Comparison experiments

Run all configured segmentation baselines and temporal models:

```bash
python scripts/run_all_comparisons.py --config configs/compare_models.yaml
```

Evaluate existing best checkpoints only:

```bash
python scripts/run_all_comparisons.py --config configs/compare_models.yaml --eval-only
```

Resume unfinished training:

```bash
python scripts/run_all_comparisons.py --config configs/compare_models.yaml --resume
```

Run selected models:

```bash
python scripts/run_all_comparisons.py --config configs/compare_models.yaml --models unet deeplabv3plus siamese_stpnet
```

Outputs are saved under `run/comparison/`. See `README_comparison.md` for the full comparison framework notes.

