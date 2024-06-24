


seeds=(12 22 32 42)
# Need to replace the path with your own path
DATA_ROOT=/data/kyungbok/FakeAVCeleb_v1.2/
OUTPUT=/data/kyungbok/outputs

for random_seed in ${seeds[@]}; do

  # MRDF margin
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type MRDF_Margin \
    --name MRDF_Margin \
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --random_seed $random_seed \
    --wandb 

  #  Dissonance
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type Dissonance \
    --name Dissonance\
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --epochs 100 \
    --random_seed $random_seed \
    --wandb \


  #  MRDF_CE
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type MRDF_CE \
    --name MRDF_CE \
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --random_seed $random_seed \
    --wandb


  # AVDF
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type AVDF \
    --name AVDF \
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --random_seed $random_seed \
    --wandb

  # AVDF MultiLabel
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type AVDF_Multilabel \
    --name AVDF_Multilabel \
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --random_seed $random_seed \
    --wandb


# AVOC Resnet
  CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type AVOC \
  --name AVOC\
  --data_root $DATA_ROOT \
  --output $OUTPUT \
  --random_seed $random_seed \
  --wandb \

  # AVOC Scnet
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type AVOC \
    --name AVOC_scnet\
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --wandb \
    --random_seed $random_seed \
    --scnet \

#  AVOC ResNet no OC 
  CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type AVOC \
  --name AVOC_no\
  --data_root $DATA_ROOT \
  --output $OUTPUT \
  --oc_option no \
  --random_seed $random_seed \
  --wandb 


# AVOC Scnet no OC
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type AVOC \
    --name AVOC_no_scnet\
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --oc_option no \
    --wandb \
    --random_seed $random_seed \
    --scnet \



  #  MSOC Resnet
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type MSOC \
    --name MSOC \
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --random_seed $random_seed \
    --wandb \

    

  # MSOC Scnet
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type MSOC \
    --name MSOC_scnet\
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --random_seed $random_seed \
    --scnet \
    --wandb \

done







