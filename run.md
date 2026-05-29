# 训练
python train_st_prompt_net_v2_BDF.py --train_dir "D:\datasets\pmtm\TBUT_Seg_Data_v1\train" --val_dir "D:\datasets\pmtm\TBUT_Seg_Data_v1\val" --save_dir "./runs/st_bdf_v1_421" --window_size 5 --img_size 512  --epochs 80 --lr 1e-4 --early_stopping 30 --amp  --num_workers 4  --net_name Sia-prompt-BDF-net

# train_ablation.py
  python run_ablation.py --train_dir "D:\datasets\pmtm\TBUT_Seg_Data_v1\train" --val_dir "D:\datasets\pmtm\TBUT_Seg_Data_v1\val" --base_save_dir ./runs/ablation_study
