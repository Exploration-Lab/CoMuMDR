export CUBLAS_WORKSPACE_CONFIG=:16:8 # to eliminate LSTM non-deterministic

if [[ "$1" == "--link_only" ]]
then
    prefix=comumdr_link_only_checkpoints_roberta
else
    prefix=comumdr_checkpoints_roberta
fi

GPU_ID=2 

for seed in 1
do
  for lr in 2e-5
  do
    for epochs in 20
    do
        python main.py --encoder_model "$prefix"/"$seed"_"$lr"_"$epochs"/ \
          --seed $seed \
          --gpu $GPU_ID \
          --data_dir data/comumdr_v2 \
          --max_num_test_contexts 37 \
          --eval \
          --fp16 \
          $1
    done
  done
done
