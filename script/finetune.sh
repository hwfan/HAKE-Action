#!/bin/bash
mkdir -p logs/$1
if [ ! -n "$2" ]
then
  iter=2000000
else
  iter=$2
fi
python -u tools/Train_pasta_HICO_DET.py --data 0 --init_weight 2 --train_module 1 --num_iteration $iter --model $1 2>&1|tee logs/$1/$1-finetune.log
