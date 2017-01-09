#!/bin/bash


###################################################################
###################################################################
############################# RGB #################################
###################################################################
###################################################################

############################ rgb variable I ############################
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_10_8bits nyud_v2_trainvalGupta_rgb_i_10_8bits nyud_v2_testGupta_rgb_i_10_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_20_8bits nyud_v2_trainvalGupta_rgb_i_20_8bits nyud_v2_testGupta_rgb_i_20_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_30_8bits nyud_v2_trainvalGupta_rgb_i_30_8bits nyud_v2_testGupta_rgb_i_30_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_40_8bits nyud_v2_trainvalGupta_rgb_i_40_8bits nyud_v2_testGupta_rgb_i_40_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_50_8bits nyud_v2_trainvalGupta_rgb_i_50_8bits nyud_v2_testGupta_rgb_i_50_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_60_8bits nyud_v2_trainvalGupta_rgb_i_60_8bits nyud_v2_testGupta_rgb_i_60_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_70_8bits nyud_v2_trainvalGupta_rgb_i_70_8bits nyud_v2_testGupta_rgb_i_70_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_80_8bits nyud_v2_trainvalGupta_rgb_i_80_8bits nyud_v2_testGupta_rgb_i_80_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_90_8bits nyud_v2_trainvalGupta_rgb_i_90_8bits nyud_v2_testGupta_rgb_i_90_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_100_8bits nyud_v2_trainvalGupta_rgb_i_100_8bits nyud_v2_testGupta_rgb_i_100_8bits 20000
############################ rgb mixed I ############################
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_r_50_10_8bits nyud_v2_trainvalGupta_i_r_50_10_rgb_i_r_50_10_8bits nyud_v2_testGupta_i_r_50_10_rgb_i_r_50_10_8bits 60000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_r_100_60_8bits nyud_v2_trainvalGupta_i_r_100_60_rgb_i_r_100_60_8bits nyud_v2_testGupta_i_r_100_60_rgb_i_r_100_60_8bits 60000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_rgb_i_all_8bits nyud_v2_trainvalGupta_i_all_rgb_i_all_8bits nyud_v2_testGupta_i_all_rgb_i_all_8bits 100000
