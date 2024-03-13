

# MSOC threshold
  CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_type MSOC \
  --name MSOC_threshold_sync \
  --dataset_type new \
  --use_threshold \
  --audio_threshold 0.5 \
  --visual_threshold 0.5 \
  --final_threshold 0.5 \
  --ckpt MSOC/new/MSOC_MSOC_threshold_sync_seed:42_epoch=4-val_loss=9.219.ckpt

# MSOC threshold
  CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_type MSOC \
  --name MSOC_threshold_sync_aug \
  --dataset_type new \
  --use_threshold \
  --audio_threshold 0.5 \
  --visual_threshold 0.5 \
  --final_threshold 0.5 \
  --ckpt MSOC/new/MSOC_MSOC_threshold_sync_aug_seed:42_epoch=8-val_loss=8.678.ckpt

# MSOC 
  CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_type MSOC \
  --name MSOC \
  --dataset_type new \
  --ckpt MSOC/new/MSOC_MSOC_seed:42_epoch=16-val_loss=8.953.ckpt


# MSOC threshold
  CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_type MSOC \
  --name MSOC_threshold \
  --dataset_type new \
  --use_threshold \
  --audio_threshold 0.5 \
  --visual_threshold 0.5 \
  --final_threshold 0.5 \
  --ckpt MSOC/new/MSOC_MSOC_threshold_seed:42_epoch=14-val_loss=9.024.ckpt

# MRDF_margin 
  CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_type MRDF_Margin \
  --name MSOC_Margin \
  --dataset_type new \
  --ckpt MRDF_Margin/new/MRDF_Margin_MRDF_Margin_margin_seed:42_epoch=5-val_loss=1.200.ckpt

