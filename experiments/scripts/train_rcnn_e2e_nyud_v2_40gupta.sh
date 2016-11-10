#!/bin/bash
# Usage:
# ./experiments/scripts/train_rcnn_e2e.sh
#
# Example:
# ./experiments/scripts/train_rcnn_e2e_nyud_v2_40gupta.sh 0 key_rgb nyud_v2_trainval_rgb nyud_v2_test_rgb 20

set -x
set -e

export PYTHONUNBUFFERED="True"


GPU_ID=$1
KEY=$2
NAME_DATASET="nyud_v2_40gupta"
NAME_TRAIN_IMDB=$3
NAME_TEST_IMDB=$4
MAX_ITERS=$5

EXP_DIR="output/${KEY}_${NAME_DATASET}_`date +'%Y-%m-%d_%H-%M-%S'`/"
mkdir -p ${EXP_DIR}
LOG_FILE="${EXP_DIR}log.txt"
exec &> >(tee -a "$LOG_FILE")


# Stage 1 Faste R-CNN e2e
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt"
PATH_BINARY="data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/cfg_end2end.yml"
OUTPUT_FILE_NAME_1="1_e2e_net.caffemodel"

./tools/jg_net_train.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_1}\
  --max_iters ${MAX_ITERS}


# stage 2.1 Compute test (no evaluation) : generate test outputs
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt"
PATH_BINARY="${EXP_DIR}${OUTPUT_FILE_NAME_1}"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/cfg_end2end.yml"
OUTPUT_FILE_NAME_21="21_net_outputs.pkl"

./tools/jg_net_generate.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TEST_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_21}




# stage 2.2 Eval test from net test outputs
mkdir -p "${EXP_DIR}results"

./tools/jg_net_evaluate.py \
  --imdb_name ${NAME_TEST_IMDB} \
  --output_dir ${EXP_DIR}\
  --input_file_name ${OUTPUT_FILE_NAME_21}
