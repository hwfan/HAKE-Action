#!/bin/bash
mkdir -p logs/$1
if [ ! -n "$2" ]
then
  iter=2000000
else
  iter=$2
fi
python -u tools/Train_pasta_HICO_DET.py --data 1 --init_weight 1 --train_module 2 --num_iteration $iter --model $1 2>&1|tee $1-train.log