#EXP_DIR="output/${KEY}/"
EXP_DIR="output/ao_d_raw_DNA_8bits_nyud_v2_40gupta_2017-01-10_00-05-58/"
TEST_IMDB="nyud_v2_testGupta_d_raw_DNA_8bits"
LOG_FILE="${EXP_DIR}results.txt"
exec &> >(tee -a "$LOG_FILE")

./tools/jg_net_generate.py \
 --gpu_id 0 \
  --path_net_proto models/nyud_v2_40gupta/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --path_net_weights ${EXP_DIR}23_net.caffemodel \
  --imdb_name ${TEST_IMDB} \
  --path_cfg experiments/cfgs/nyud_v2_40gupta/netTest.yml \
  --output_dir ${EXP_DIR} \
  --output_file_name 31_net_outputs.pkl

mkdir -p "${EXP_DIR}results"
./tools/jg_net_evaluate.py \
    --imdb_name ${TEST_IMDB} \
    --output_dir ${EXP_DIR} \
    --input_file_name 31_net_outputs.pkl
