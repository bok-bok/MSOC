


TEST_SUBSETS=(C D E F)
# PRED_STRATEGIES=(mean min)
PRED_STRATEGIES=(mean)

# Need to replace the path with your own path
DATA_ROOT=/data/kyungbok/FakeAVCeleb_v1.2/
OUTPUT=/data/kyungbok/outputs

file_name=_final_all

random_seeds=(12 22 32 42)

for random_seed in ${random_seeds[@]}; do
    for test_subset in ${TEST_SUBSETS[@]}; do

        # AVOC no oc scnet
        CUDA_VISIBLE_DEVICES=0 python eval_all.py \
        --model_type AVOC \
        --data_root $DATA_ROOT \
        --output $OUTPUT \
        --file_name $file_name \
        --test_subset $test_subset \
        --random_seed $random_seed \
        --oc_option no \
        --scnet \

        # AVOC no oc 
        CUDA_VISIBLE_DEVICES=0 python eval_all.py \
        --model_type AVOC \
        --data_root $DATA_ROOT \
        --output $OUTPUT \
        --file_name $file_name \
        --test_subset $test_subset \
        --random_seed $random_seed \
        --oc_option no \
        --light \



        # AVOC
        CUDA_VISIBLE_DEVICES=0 python eval_all.py \
        --model_type AVOC \
        --data_root $DATA_ROOT \
        --output $OUTPUT \
        --file_name $file_name \
        --test_subset $test_subset \
        --random_seed $random_seed \
        --light





        # AVOC scnet
        CUDA_VISIBLE_DEVICES=0 python eval_all.py \
        --model_type AVOC \
        --data_root $DATA_ROOT \
        --output $OUTPUT \
        --file_name $file_name \
        --test_subset $test_subset \
        --random_seed $random_seed \
        --scnet \


    # Multimodal dissonance
        CUDA_VISIBLE_DEVICES=0 python eval_all.py \
        --model_type Dissonance \
        --data_root $DATA_ROOT \
        --output $OUTPUT \
        --file_name $file_name \
        --test_subset $test_subset \
        --random_seed $random_seed



        # avdf_multilabel
        CUDA_VISIBLE_DEVICES=0 python eval_all.py \
        --model_type AVDF_Multilabel \
        --data_root $DATA_ROOT \
        --output $OUTPUT \
        --file_name $file_name \
        --test_subset $test_subset \
        --random_seed $random_seed

    # MRDF_margin 
        CUDA_VISIBLE_DEVICES=0 python eval_all.py \
        --model_type MRDF_Margin \
        --data_root $DATA_ROOT \
        --output $OUTPUT \
        --file_name $file_name \
        --test_subset $test_subset \
        --random_seed $random_seed


    # MRDF CE
        CUDA_VISIBLE_DEVICES=0 python eval_all.py \
        --model_type MRDF_CE \
        --data_root $DATA_ROOT \
        --output $OUTPUT \
        --file_name $file_name \
        --test_subset $test_subset \
        --random_seed $random_seed



# AVDF
    CUDA_VISIBLE_DEVICES=0 python eval_all.py \
    --model_type AVDF \
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --file_name $file_name \
    --random_seed $random_seed \
    --test_subset $test_subset \





    # for pred_stred in ${PRED_STRATEGIES[@]}; do
    # TalkNet 
    CUDA_VISIBLE_DEVICES=0 python eval_all.py \
    --model_type MSOC \
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --test_subset $test_subset \
    --random_seed $random_seed \
    --save_score \
    --file_name $file_name \

    # TalkNet SF scnet 
    CUDA_VISIBLE_DEVICES=0 python eval_all.py \
    --model_type MSOC \
    --data_root $DATA_ROOT \
    --output $OUTPUT \
    --test_subset $test_subset \
    --random_seed $random_seed \
    --save_score \
    --file_name $file_name \
    --scnet \

done


done