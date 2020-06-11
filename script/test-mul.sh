#!/bin/bash
mkdir -p logs/$2
if [ ! -n "$3" ]
then
  iter=2000000
else
  iter=$3
fi
python -u tools/Test_mulproc_HICO_DET.py --gpu $1 --iteration $iter --model $2 2>&1|tee logs/$2/$2-test.log
pushd ./-Results/
python Generate_detection.py --model "$iter"_"$2"
python Evaluate_HICO_DET.py --file Detection_"$iter"_"$2.pkl" 2>&1|tee ../logs/$2/$2-result.log
popd