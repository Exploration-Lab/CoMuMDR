export CUBLAS_WORKSPACE_CONFIG=:16:8 # to eliminate LSTM non-deterministic
rm -rf *convin*.pkl # fresh start

if [[ "$1" == "--link_only" ]]
then
    prefix=comumdr_link_only_checkpoints_roberta
else
    prefix=comumdr_checkpoints_roberta
fi

GPU_ID=1

for seed in 1
do
  for lr in 2e-5
  do
    for epochs in 20
    do
        # python main.py --encoder_model ./roberta_base \
        python main.py --encoder_model ./paraphrase_xlm_r \
          --output_dir "$prefix"/"$seed"_"$lr"_"$epochs"/ \
          --seed $seed \
          --gpu $GPU_ID \
          --data_dir data/comumdr_v2 \
          --max_num_train_contexts 20 \
          --max_num_dev_contexts 37 \
          --max_num_test_contexts 37 \
          --num_train_epochs $epochs \
          --train_batch_size 2 \
          --learning_rate $lr \
          --use_scheduler \
          --fp16 \
          $1
    done
  done
done
