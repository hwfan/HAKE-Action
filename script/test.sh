#!/bin/bash
mkdir -p logs/$1
if [ ! -n "$2" ]
then
  iter=2000000
else
  iter=$2
fi
python -u tools/Test_pasta_HICO_DET.py --iteration $iter --model $1 2>&1|tee $1-test.log