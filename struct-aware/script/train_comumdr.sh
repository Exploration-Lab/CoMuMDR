#!/usr/bin/env bash
train_file="comumdr_data/train.json"
eval_file="comumdr_data/dev.json"
test_file="comumdr_data/test.json"
glove_file="./glove.6B.200d.txt"
# dataset_dir="../dataset/Molweni"
dataset_dir="./dataset/comumdr_electra"
model_dir="./convin"

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi

GPU=0
model_name=model
task=student  # student, teacher or distill

CUDA_VISIBLE_DEVICES=${GPU} python main_electra.py --train_file=$train_file --eval_file=$eval_file --test_file=$test_file \
                                    --dataset_dir=$dataset_dir --glove_vocab_path $glove_file \
                                    --epoches 30 --batch_size 100 --pool_size 100 \
                                    --eval_pool_size 10 --report_step 30 \
                                    --save_model --overwrite --do_train \
                                    --model_path "${model_dir}/${model_name}.pt" \
                                    --teacher_model_path "${model_dir}/teacher_model.pt" \
                                    --num_layers 3 --remake_dataset \
                                    --task ${task} > ${paraphrase_xlm_r}.log 2>&1 &

#CUDA_VISIBLE_DEVICES=${GPU} nohup python -u main.py --train_file=$train_file --eval_file=$eval_file --test_file=$test_file \
#                                    --dataset_dir=$dataset_dir --glove_vocab_path $glove_file \
#                                    --epoches 30 --batch_size 100 --pool_size 100 \
#                                    --eval_pool_size 10 --report_step 30 \
#                                    --save_model --overwrite --do_train \
#                                    --model_path "${model_dir}/${model_name}.pt" \
#                                    --teacher_model_path "${model_dir}/teacher_model.pt" \
#                                    --num_layers 3 --remake_dataset \
#                                    --task ${task} > ${model_name}.log 2>&1 &