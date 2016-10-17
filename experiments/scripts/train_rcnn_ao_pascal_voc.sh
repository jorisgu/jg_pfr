#!/bin/bash
# Usage:
# ./experiments/scripts/train_rcnn_ao.sh
#
# Example:
# ./experiments/scripts/train_rcnn_ao.sh

set -x
set -e

export PYTHONUNBUFFERED="True"

KEY="myscript4"
GPU_ID=0
NAME_DATASET="pascal_voc"
NAME_TRAIN_IMDB="voc_2007_trainval"
NAME_TEST_IMDB="voc_2007_test"

EXP_DIR="output/${KEY}_${NAME_DATASET}/" #_`date +'%Y-%m-%d_%H-%M-%S'`/"
mkdir -p ${EXP_DIR}
LOG_FILE="${EXP_DIR}log.txt"
exec &> >(tee -a "$LOG_FILE")


# Stage 1.1 RPN, init from ImageNet model
PATH_PROTO="models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_alt_opt/stage1_rpn_solver60k80k.pt"
PATH_BINARY="data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel"
PATH_CONFIG_FILE="experiments/cfgs/pascal_voc/rpnTrain.yml"
MAX_ITERS=100
OUTPUT_FILE_NAME="1_rpn.caffemodel"

./tools/jg_rpn_train.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME}\
  --max_iters ${MAX_ITERS}


# Stage 1.2 RPN, generate proposals
PATH_PROTO="models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_alt_opt/rpn_test.pt"
PATH_BINARY="${EXP_DIR}${OUTPUT_FILE_NAME}"
PATH_CONFIG_FILE="experiments/cfgs/pascal_voc/rpnGenerate.yml"
OUTPUT_FILE_NAME="1_rpn_proposals.pkl"

./tools/jg_rpn_generate.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME}


# Stage 1.3 Fast R-CNN using RPN proposals, init from ImageNet model
PATH_PROTO="models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_alt_opt/stage1_fast_rcnn_solver30k40k.pt"
PATH_BINARY="data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel"
PATH_CONFIG_FILE="experiments/cfgs/pascal_voc/netTrain.yml"
MAX_ITERS=100
PATH_PROPOSAL="${EXP_DIR}${OUTPUT_FILE_NAME}"
OUTPUT_FILE_NAME="1_net.caffemodel"


./tools/jg_net_train.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME}\
  --path_proposal ${PATH_PROPOSAL}\
  --max_iters ${MAX_ITERS}
# Stage 2.1 RPN, init from stage 1 Fast R-CNN model
# Stage 2.2 RPN, generate proposals
# Stage 2.3 Fast R-CNN, init from stage 2 RPN R-CNN model
