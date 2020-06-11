#!/bin/bash

# ---------------HICO-DET Dataset------------------
echo "Downloading Dataset"

mkdir Data/
mkdir Weights/
ddl 'https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk' Data/hico_20160224_det.tar.gz --proxy socks5h://127.0.0.1:1080
tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
rm Data/hico_20160224_det.tar.gz

echo "Downloading training data..."

ddl 'https://drive.google.com/open?id=1uhtAg00EGZelc6mhEowQ3vwxWN1YBmXC' Data/Trainval_GT_all_part.pkl --proxy socks5h://127.0.0.1:1080
ddl 'https://drive.google.com/open?id=1cqaJxzFRrMENTYGcuZFYy8JRuTP_qAEq' Data/Trainval_Neg_all_part.pkl --proxy socks5h://127.0.0.1:1080
ddl 'https://drive.google.com/open?id=1KnQvUklj8p_cld1FagsUK6BEUeHqBtpR' Data/Test_all_part.pkl --proxy socks5h://127.0.0.1:1080
ddl 'https://drive.google.com/open?id=1hkakTuYB_3C4GCbZgSm2pH7AHbhmEr5y' Data/ava_train_all_fixed.pkl --proxy socks5h://127.0.0.1:1080
ddl 'https://drive.google.com/open?id=1cYR2rI8H78sY7exv-L3HXYhXqp7HQdUU' Data/ava_val_fixed.pkl --proxy socks5h://127.0.0.1:1080
ddl 'https://drive.google.com/open?id=1ew-uIrD_ZFmO5RFsuMV2sBEqWWVtXkHH' Weights/pasta_full.zip --proxy socks5h://127.0.0.1:1080
ddl 'https://drive.google.com/open?id=1h0InSjxCffLuyoXEPbLIzOH0KVWUuUJc' Weights/pasta_AVA.zip --proxy socks5h://127.0.0.1:1080
ddl 'https://drive.google.com/open?id=1l12Dr217NeTnbBA5e-9lLetMFmDv7eI9' Weights/pasta_HICO-DET.zip --proxy socks5h://127.0.0.1:1080
ddl 'https://drive.google.com/open?id=1_O8eo1fmJtVzb_W5Y4emg_eE7LTJKPGR' Results.tar.gz --proxy socks5h://127.0.0.1:1080
ddl 'https://drive.google.com/open?id=1_ubvIeBVIBT2-M0taN4ytf72B5xx0JR7' lib/ult/matrix_sentence_76.py --proxy socks5h://127.0.0.1:1080
tar -xzvf Results.tar.gz
rm Results.tar.gz
ln ./-Results/HICO_DET_utils.py ./lib/ult/HICO_DET_utils.py

ddl 'https://1drv.ms/u/s!ArUVoRxpBphY1jwevFWuaSetd4ay?e=y9xBRY' Weights/res50_faster_rcnn_iter_1190000.ckpt.data-00000-of-00001 --proxy socks5h://127.0.0.1:1080
ddl 'https://1drv.ms/u/s!ArUVoRxpBphY1jpA3iVqOsMdISvl?e=3MXJIZ' Weights/res50_faster_rcnn_iter_1190000.ckpt.index --proxy socks5h://127.0.0.1:1080
ddl 'https://1drv.ms/u/s!ArUVoRxpBphY1juAa4suqiYZGujt?e=hrWnEu' Weights/res50_faster_rcnn_iter_1190000.ckpt.meta --proxy socks5h://127.0.0.1:1080