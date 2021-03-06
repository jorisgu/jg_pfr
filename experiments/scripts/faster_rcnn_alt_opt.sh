#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_alt_opt.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is only pascal_voc for now
#
# Example:
# ./experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"


set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    #TRAIN_IMDB="voc_2007_trainval"
    TRAIN_IMDB="voc_2007_fake"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=40000
    ;;
  nyud_v2_rgb)
    TRAIN_IMDB="nyud_v2_trainval_rgb"
    TEST_IMDB="nyud_v2_test_rgb"
    PT_DIR="nyud_v2"
    ITERS=20000
    ;;
  nyud_v2)
    TRAIN_IMDB="nyud_v2_trainval_d_raw_normal_16bits"
    TEST_IMDB="nyud_v2_test_d_raw_normal_16bits"
    PT_DIR="nyud_v2"
    ITERS=20000
    ;;
  coco)
    echo "Not implemented: use experiments/scripts/faster_rcnn_end2end.sh for coco"
    exit
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/`date +'%Y-%m-%d_%H-%M-%S'`.pfr_ao_${NET}_${EXTRA_ARGS_SLUG}.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_faster_rcnn_alt_opt.py --gpu ${GPU_ID} \
  --net_name ${NET} \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --basic_db ${DATASET} \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}


  #./tools/test_net.py --gpu 0 --def models/nyud_v2/VGG_CNN_M_1024/faster_rcnn_alt_opt/faster_rcnn_test.pt --net /data/workspace/jg_pfr/output/test_nyudv2_fake2/nyud_v2_val_rgb/VGG_CNN_M_1024_faster_rcnn_final.caffemodel --imdb nyud_v2_test_rgb --cfg experiments/cfgs/faster_rcnn_alt_opt.yml --set EXP_DIR test_nyudv2_fake2 RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"
