#!/bin/bash
# Example:
# ./experiments/scripts/thescript.sh folder imdb
 #./experiments/scripts/thescript.sh



set -x
set -e

export PYTHONUNBUFFERED="True"

CANAL=$1
NAME_TEST_IMDB=$2

EXP_DIR="output/all_results_voc2010/${CANAL}/${SET}/"
mkdir -p ${EXP_DIR}
LOG_FILE="${EXP_DIR}log.txt"
exec &> >(tee -a "$LOG_FILE")

# stage 1.1 Compute test (no evaluation) : generate test outputs
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_alt_opt/faster_rcnn_test.pt"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/netTest.yml"
OUTPUT_FILE_NAME="bb_generation.pkl"

./tools/jg_net_generate.py \
  --gpu_id 0 \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights output/${CANAL}/23_net.caffemodel \
  --imdb_name ${NAME_TEST_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR} \
  --output_file_name ${OUTPUT_FILE_NAME}




# stage 3.2 Eval test from net test outputs
mkdir -p "${EXP_DIR}results"

./tools/jg_net_evaluate.py \
  --imdb_name ${NAME_TEST_IMDB} \
  --output_dir ${EXP_DIR}\
  --input_file_name ${OUTPUT_FILE_NAME}
