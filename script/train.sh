#!/bin/bash
mkdir -p logs/$2
if [ ! -n "$3" ]
then
  iter=2000000
else
  iter=$3
fi
python -u tools/Train_pasta_HICO_DET.py --gpu $1 --data 0 --init_weight 1 --train_module 2 --num_iteration $iter --model $2 2>&1|tee logs/$2/$2-train.log
