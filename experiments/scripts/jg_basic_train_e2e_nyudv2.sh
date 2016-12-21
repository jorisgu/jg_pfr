#!/bin/bash
# ./experiments/scripts/jg_e2e_basic.sh gpuId experimentFolder trainIMDB testIMDB maxIters
# ./experiments/scripts/jg_e2e_basic.sh 0 e2e_test_nyudv2 voc_2007_fake voc_2007_fake 100
set -x
set -e

export PYTHONUNBUFFERED="True"


GPU_ID=$1
KEY=$2
NAME_DATASET="nyudv2"
NAME_TRAIN_IMDB=$3
NAME_TEST_IMDB=$4
MAX_ITERS=$5

EXP_DIR="output/${KEY}_${NAME_DATASET}_`date +'%Y-%m-%d_%H-%M-%S'`/"
mkdir -p ${EXP_DIR}
LOG_FILE="${EXP_DIR}log.txt"
exec &> >(tee -a "$LOG_FILE")

#PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_end2end/solver_pdconv5.prototxt"
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt"
#PATH_BINARY="/home/jguerry/workspace/jg_pfr/output/e2e_test_nyudv2_nyudv2_2016-12-21_17-13-14/net.caffemodel"
PATH_BINARY="data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/cfg_end2end.yml"
OUTPUT_FILE_NAME_1="net.caffemodel"

./tools/jg_net_train.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TRAIN_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_1}\
  --max_iters ${MAX_ITERS}



# generate output for test
PATH_PROTO="models/nyud_v2_40gupta/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt"
PATH_BINARY="${EXP_DIR}${OUTPUT_FILE_NAME_1}"
PATH_CONFIG_FILE="experiments/cfgs/nyud_v2_40gupta/netTest.yml"
OUTPUT_FILE_NAME_21="outputs.pkl"

./tools/jg_net_generate.py \
  --gpu_id ${GPU_ID} \
  --path_net_proto ${PATH_PROTO} \
  --path_net_weights ${PATH_BINARY} \
  --imdb_name ${NAME_TEST_IMDB} \
  --path_cfg ${PATH_CONFIG_FILE} \
  --output_dir ${EXP_DIR}\
  --output_file_name ${OUTPUT_FILE_NAME_21}


mkdir -p "${EXP_DIR}results"

./tools/jg_net_evaluate.py \
  --imdb_name ${NAME_TEST_IMDB} \
  --output_dir ${EXP_DIR}\
  --input_file_name ${OUTPUT_FILE_NAME_21}

    #./tools/jg_net_evaluate.py \
    #  --imdb_name nyud_v2_fake_rgb_raw_8bits \
    #  --output_dir output/ao_rgb_raw_8bits_nyud_v2_40gupta_2016-11-22_13-36-28/ \
    #  --input_file_name 31_net_outputs.pkl
