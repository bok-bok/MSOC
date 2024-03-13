

# python new_main.py --margin_type margin

# python new_main.py --margin_type oc --alpha 0 

# python new_main.py --margin_type oc --alpha 20

# train MRDF_Margin




# #  MRDF_Margin - margin
#   CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
#   --model_type MRDF_Margin \
#   --name MRDF_Margin_margin \
#   --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
#   --dataset fakeavceleb \
#   --max_frames 30 \
#   --dataset_type new \
#   --wandb






# # MSOC_Sync
#   CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
#   --model_type MSOC_Sync \
#   --name MSOC_Sync \
#   --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
#   --dataset fakeavceleb \
#   --margin_contrast -1.0 \
#   --max_frames 30 \
#   --dataset_type new \
#   --wandb


# # #  MRDF_CE
# #   CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
# #   --model_type MRDF_CE \
# #   --name MRDF_CE \
# #   --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
# #   --dataset fakeavceleb \
# #   --max_frames 30 \
# #   --dataset_type new \
# #   --wandb

# # MSOC_Sync_aug
#   CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
#   --model_type MSOC_Sync \
#   --name MSOC_Sync_aug \
#   --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
#   --dataset fakeavceleb \
#   --margin_contrast -1.0 \
#   --max_frames 30 \
#   --dataset_type new \
#   --wandb \
#   --augmentation \


#   # # MRDF_Margin - oc_no_contrast
#   # CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
#   # --model_type MRDF_Margin_OC \
#   # --name MRDF_Margin_oc_no_contrast \
#   # --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
#   # --dataset fakeavceleb \
#   # --margin_contrast -1.0 \
#   # --max_frames 30 \
#   # --dataset_type new \
#   # --wandb



# #  MSOC
#   CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
#   --model_type MSOC \
#   --name MSOC \
#   --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
#   --dataset fakeavceleb \
#   --margin_contrast -1.0 \
#   --max_frames 30 \
#   --dataset_type new \
#   --wandb

# MRDF_Margin - oc
  # CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
  # --model_type MRDF_Margin_OC \
  # --name MRDF_Margin_oc \
  # --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
  # --dataset fakeavceleb \
  # --max_frames 30 \
  # --dataset_type new \
  # --wandb



# #  MSOC
#   CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
#   --model_type MSOC \
#   --name MSOC \
#   --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
#   --dataset fakeavceleb \
#   --margin_contrast -1.0 \
#   --max_frames 30 \
#   --dataset_type new \
#   --wandb

# # MSOC threshold
#   CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
#   --model_type MSOC \
#   --name MSOC_threshold \
#   --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
#   --dataset fakeavceleb \
#   --max_frames 30 \
#   --dataset_type new \
#   --use_threshold \
#   --wandb

# MSOC sync threshold
  CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
  --model_type MSOC \
  --name MSOC_threshold_sync \
  --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
  --dataset fakeavceleb \
  --max_frames 30 \
  --dataset_type new \
  --use_threshold \
  --sync \
  --wandb

  # MSOC sync threshold
  CUDA_VISIBLE_DEVICES=0 python train_light_new.py \
  --model_type MSOC \
  --name MSOC_threshold_sync_aug \
  --data_root /data/kyungbok/FakeAVCeleb_v1.2/ \
  --dataset fakeavceleb \
  --max_frames 30 \
  --dataset_type new \
  --use_threshold \
  --sync \
  --augmentation \
  --wandb