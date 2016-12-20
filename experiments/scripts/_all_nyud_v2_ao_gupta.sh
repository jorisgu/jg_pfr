#!/bin/bash


###################################################################
###################################################################
############################# RGB #################################
###################################################################
###################################################################

############################ rgb raw ############################
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_100_8bits nyud_v2_trainvalGupta_rgb_i_100_8bits nyud_v2_testGupta_rgb_i_100_8bits 20000

############################ rgb variable I ############################
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_10_8bits nyud_v2_trainvalGupta_rgb_i_10_8bits nyud_v2_testGupta_rgb_i_10_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_20_8bits nyud_v2_trainvalGupta_rgb_i_20_8bits nyud_v2_testGupta_rgb_i_20_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_30_8bits nyud_v2_trainvalGupta_rgb_i_30_8bits nyud_v2_testGupta_rgb_i_30_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_40_8bits nyud_v2_trainvalGupta_rgb_i_40_8bits nyud_v2_testGupta_rgb_i_40_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_20000_8bits nyud_v2_trainvalGupta_rgb_i_20000_8bits nyud_v2_testGupta_rgb_i_20000_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_60_8bits nyud_v2_trainvalGupta_rgb_i_60_8bits nyud_v2_testGupta_rgb_i_60_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_70_8bits nyud_v2_trainvalGupta_rgb_i_70_8bits nyud_v2_testGupta_rgb_i_70_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_80_8bits nyud_v2_trainvalGupta_rgb_i_80_8bits nyud_v2_testGupta_rgb_i_80_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_90_8bits nyud_v2_trainvalGupta_rgb_i_90_8bits nyud_v2_testGupta_rgb_i_90_8bits 20000

############################ rgb mixed ############################
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_r_20000_10_8bits nyud_v2_trainvalGupta_i_r_20000_10_rgb_i_r_20000_10_8bits nyud_v2_testGupta_i_r_20000_10_rgb_i_r_20000_10_8bits 60000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_r_100_60_8bits nyud_v2_trainvalGupta_i_r_100_60_rgb_i_r_100_60_8bits nyud_v2_testGupta_i_r_100_60_rgb_i_r_100_60_8bits 60000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_rgb_i_all_8bits nyud_v2_trainvalGupta_i_all_rgb_i_all_8bits nyud_v2_testGupta_i_all_rgb_i_all_8bits 100000


./experiments/scripts/_computeResults_rgbIntensity_all_nyud_v2_ao.sh


###################################################################
###################################################################
############################# DEPTH ###############################
###################################################################
###################################################################


############################ depth raw ############################
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_normal_8bits nyud_v2_trainvalGupta_d_raw_normal_8bits nyud_v2_testGupta_d_raw_normal_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_normal_16bits nyud_v2_trainvalGupta_d_raw_normal_16bits nyud_v2_testGupta_d_raw_normal_16bits 20000

############################ focus 95% ############################
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_focus_8bits nyud_v2_trainvalGupta_d_raw_focus_8bits nyud_v2_testGupta_d_raw_focus_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_focus_16bits nyud_v2_trainvalGupta_d_raw_focus_16bits nyud_v2_testGupta_d_raw_focus_16bits 20000

############################ HHA ############################
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_HHA_8bits nyud_v2_trainvalGupta_d_raw_HHA_8bits nyud_v2_testGupta_d_raw_HHA_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_HHA_focus_8bits nyud_v2_trainvalGupta_d_raw_HHA_focus_8bits nyud_v2_testGupta_d_raw_HHA_focus_8bits 20000

############################ jet ############################
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_jet_8bits nyud_v2_trainvalGupta_d_raw_jet_8bits nyud_v2_testGupta_d_raw_jet_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_jet_focus_8bits nyud_v2_trainvalGupta_d_raw_jet_focus_8bits nyud_v2_testGupta_d_raw_jet_focus_8bits 20000

############################ cubehelix ############################
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_cubehelix_8bits nyud_v2_trainvalGupta_d_raw_cubehelix_8bits nyud_v2_testGupta_d_raw_cubehelix_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_cubehelix_focus_8bits nyud_v2_trainvalGupta_d_raw_cubehelix_focus_8bits nyud_v2_testGupta_d_raw_cubehelix_focus_8bits 20000

############################ histeq ############################
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_histeqRandom_8bits nyud_v2_trainvalGupta_d_raw_histeqRandom_8bits nyud_v2_testGupta_d_raw_histeqRandom_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_histeqBack_8bits nyud_v2_trainvalGupta_d_raw_histeqBack_8bits nyud_v2_testGupta_d_raw_histeqBack_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_histeqFront_8bits nyud_v2_trainvalGupta_d_raw_histeqFront_8bits nyud_v2_testGupta_d_raw_histeqFront_8bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_histeqRandom_16bits nyud_v2_trainvalGupta_d_raw_histeqRandom_16bits nyud_v2_testGupta_d_raw_histeqRandom_16bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_histeqBack_16bits nyud_v2_trainvalGupta_d_raw_histeqBack_16bits nyud_v2_testGupta_d_raw_histeqBack_16bits 20000
./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh 0 ao_d_raw_histeqFront_16bits nyud_v2_trainvalGupta_d_raw_histeqFront_16bits nyud_v2_testGupta_d_raw_histeqFront_16bits 20000
