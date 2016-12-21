#!/bin/bash
# ./experiments/scripts/jg_e2e_basic.sh gpuId experimentFolder trainIMDB testIMDB maxIters
# ./experiments/scripts/jg_e2e_basic.sh 0 e2e_test_nyudv2 voc_2007_fake voc_2007_fake 100
set -x
set -e

export PYTHONUNBUFFERED="True"


GPU_ID=$1
KEY=$2
NAME_DATASET="pascal_voc"
NAME_TRAIN_IMDB=$3
NAME_TEST_IMDB=$4
MAX_ITERS=$5

EXP_DIR="output/${KEY}_${NAME_DATASET}_`date +'%Y-%m-%d_%H-%M-%S'`/"
mkdir -p ${EXP_DIR}
LOG_FILE="${EXP_DIR}log.txt"
exec &> >(tee -a "$LOG_FILE")

PATH_PROTO="models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt"
PATH_BINARY="data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel"
PATH_CONFIG_FILE="experiments/cfgs/faster_rcnn_end2end.yml"
OUTPUT_FILE_NAME_13="net.caffemodel"

./tools/jg_net_train.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_13}\
  --max_iters ${MAX_ITERS}
