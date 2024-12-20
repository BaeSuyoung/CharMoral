#!/bin/sh

# MAD training 
TASKS="A,SA,IA,CA,SIA,SCA,ICA,SICA"
MODEL_TYPE='bert' # roberta
MODEL_NAME='bert-large-uncased'
DATA_DIR='data/moral_stories_dataset' # data/movie_dataset

for task in ${TASKS//,/ }
do
    python3 models/MAD/classification_task.py \
        --model_type ${MODEL_TYPE} \
        --model_name_or_path ${MODEL_NAME} \
        --task_name ${task} \
        --do_eval \
        --do_prediction \
        --do_lower_case \
        --data_dir ${DATA_DIR} \
        --max_seq_length 100 \
        --per_gpu_eval_batch_size 8 \
        --per_gpu_train_batch_size 8 \
        --learning_rate 1e-5 \
        --num_train_epochs 50 \
        --output_dir output \
        --do_train \
        --logging_steps 500 \
        --save_steps 500 \
        --seed 32 \
        --data_cache_dir cache \
        --warmup_pct 0.1 \
        --evaluate_during_training \
        --save_total_limit 20 \
        --patience 10 \
        --overwrite_output_dir
done