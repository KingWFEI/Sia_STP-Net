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


# 版本改进实验

后续 v1-v5 版本优化实验统一采用消融 `full` 的训练 recipe，保证结构对比时训练参数一致：

- `window_size=5`
- `img_size=512`
- `batch_size=2`
- `accumulation_steps=8`
- `epochs=100`
- `lr=1e-4`
- `weight_decay=0.01`
- `early_stopping=30`
- scheduler `min_lr=1e-6`
- deep supervision 辅助头权重 `aux_weights=[0.4, 0.2, 0.1]`

```bash
# 重跑 v1-v5；--force 表示覆盖已有 best_dice.pth 结果
# 该命令使用 train_allversion.py 的默认训练参数，默认已对齐消融实验
python train_allversion.py --versions v1 v2 v3 v4 v5 --force --compute_hd95 --hd95_every 1
```

```bash
# 只跑部分版本，例如只重跑 v2 和 v4
python train_allversion.py --versions v2 v4 --force --compute_hd95 --hd95_every 1
```

```bash
# 单独跑一个版本，例如只跑 v5
python train_allversion.py --versions v5 --force --compute_hd95 --hd95_every 1
```

```bash
# 等价的完整显式命令，便于核对所有关键参数
python train_allversion.py --versions v1 v2 v3 v4 v5 --window_size 5 --img_size 512 --batch_size 2 --accumulation_steps 8 --epochs 100 --lr 1e-4 --weight_decay 0.01 --early_stopping 30 --force --compute_hd95 --hd95_every 1
```
