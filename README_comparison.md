# One-Click Comparison Experiments

Run all configured models:

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

Outputs are written to `run/comparison/`:

```text
run/comparison/
  checkpoints/
  logs/
  predictions/
  metrics/
  summary.md
```

The result table is exported as CSV, XLSX, JSON, and Markdown with:

```text
Method, Params(M), FLOPs(G), FPS, Dice, Precision, Recall, Specificity, HD95
```

All models use the same configured train/validation/test directories, `T=5`, target frame index `T//2`, `512x512` input size, AdamW, AMP, gradient accumulation, cosine annealing, and validation Dice best checkpoint selection.

Static models receive `image_seq[:, target_idx]`. Temporal models receive the full `[B,T,C,H,W]` sequence. The 2.5D U-Net is registered as temporal input and internally stacks the five frames as channels for a 2D U-Net.

