#!/bin/bash
# Usage:
# ./experiments/scripts/train_rcnn_ao.sh
#
# Example:
# ./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 rgb nyud_v2_trainval_rgb nyud_v2_test_rgb 20

set -x
set -e

export PYTHONUNBUFFERED="True"


GPU_ID=$1
KEY=$2
NAME_DATASET="nyud_v2_40gupta"
NAME_TRAIN_IMDB=$3
NAME_TEST_IMDB=$4
MAX_ITERS=$5

#EXP_DIR="output/${KEY}_${NAME_DATASET}_`date +'%Y-%m-%d_%H-%M-%S'`/"
EXP_DIR="output/${KEY}_${NAME_DATASET}/"
mkdir -p ${EXP_DIR}
LOG_FILE="${EXP_DIR}log.txt"
exec &> >(tee -a "$LOG_FILE")


# Stage 1.1 RPN, init from ImageNet model
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_alt_opt/stage1_rpn_solver60k80k.pt"
PATH_BINARY="data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/rpnTrain.yml"
OUTPUT_FILE_NAME_11="11_rpn.caffemodel"
:'
./tools/jg_rpn_train.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_11}\
  --max_iters ${MAX_ITERS}
'



# Stage 1.2 RPN, generate proposals
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_alt_opt/rpn_test.pt"
PATH_BINARY="${EXP_DIR}${OUTPUT_FILE_NAME_11}"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/rpnGenerate.yml"
OUTPUT_FILE_NAME_12="12_rpn_proposals.pkl"
:'
./tools/jg_rpn_generate.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_12}
'



# Stage 1.3 Fast R-CNN using RPN proposals, init from ImageNet model
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_alt_opt/stage1_fast_rcnn_solver30k40k.pt"
PATH_BINARY="data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/netTrain.yml"
PATH_PROPOSAL="${EXP_DIR}${OUTPUT_FILE_NAME_12}"
OUTPUT_FILE_NAME_13="13_net.caffemodel"
:'
./tools/jg_net_train.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_13}\
  --path_proposal ${PATH_PROPOSAL}\
  --max_iters ${MAX_ITERS}
:'



# Stage 2.1 RPN, init from stage 1 Fast R-CNN model
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_alt_opt/stage2_rpn_solver60k80k.pt"
PATH_BINARY="${EXP_DIR}${OUTPUT_FILE_NAME_13}"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/rpnTrain.yml"
OUTPUT_FILE_NAME_21="21_rpn.caffemodel"
:'
./tools/jg_rpn_train.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_21}\
  --max_iters ${MAX_ITERS}
:'



# Stage 2.2 RPN, generate proposals
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_alt_opt/rpn_test.pt"
PATH_BINARY="${EXP_DIR}${OUTPUT_FILE_NAME_21}"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/rpnGenerate.yml"
OUTPUT_FILE_NAME_22="22_rpn_proposals.pkl"
:'
./tools/jg_rpn_generate.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_22}
:'



# Stage 2.3 Fast R-CNN, init from stage 2 RPN R-CNN model
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_alt_opt/stage2_fast_rcnn_solver30k40k.pt"
PATH_BINARY="${EXP_DIR}${OUTPUT_FILE_NAME_21}"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/netTrain.yml"
PATH_PROPOSAL="${EXP_DIR}${OUTPUT_FILE_NAME_22}"
OUTPUT_FILE_NAME_23="23_net.caffemodel"
:'
./tools/jg_net_train.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_23}\
  --path_proposal ${PATH_PROPOSAL}\
  --max_iters ${MAX_ITERS}
:'



# stage 3.1 Compute test (no evaluation) : generate test outputs
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_alt_opt/faster_rcnn_test.pt"
PATH_BINARY="${EXP_DIR}${OUTPUT_FILE_NAME_23}"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/netTest.yml"
OUTPUT_FILE_NAME_31="31_net_outputs.pkl"

./tools/jg_net_generate.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TEST_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_31}

  #./tools/jg_net_generate.py \
  # --gpu_id 0 \
  #  --path_net_proto models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  #  --path_net_weights output/ao_rgb_raw_8bits_nyud_v2_40gupta_2016-11-22_13-36-28/23_net.caffemodel \
  #  --imdb_name nyud_v2_fake_rgb_raw_8bits \
  #  --path_cfg experiments/cfgs/nyud_v2_40gupta/netTest.yml \
  #  --output_dir output/ao_rgb_raw_8bits_nyud_v2_40gupta_2016-11-22_13-36-28/ \
  #  --output_file_name 31_net_outputs.pkl

# stage 3.2 Eval test from net test outputs
mkdir -p "${EXP_DIR}results"

./tools/jg_net_evaluate.py \
  --imdb_name ${NAME_TEST_IMDB} \
  --output_dir ${EXP_DIR}\
  --input_file_name ${OUTPUT_FILE_NAME_31}

  #./tools/jg_net_evaluate.py \
  #  --imdb_name nyud_v2_fake_rgb_raw_8bits \
  #  --output_dir output/ao_rgb_raw_8bits_nyud_v2_40gupta_2016-11-22_13-36-28/ \
  #  --input_file_name 31_net_outputs.pkl
