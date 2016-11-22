#!/bin/bash

############################ rgb raw ############################
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_rgb_raw_8bits_nyud_v2_40gupta nyud_v2_test_rgb_raw_8bits

############################ depth raw ############################
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_normal_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_normal_8bits
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_normal_16bits_nyud_v2_40gupta nyud_v2_test_d_raw_normal_16bits

############################ focus 95% ############################
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_focus_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_focus_8bits
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_focus_16bits_nyud_v2_40gupta nyud_v2_test_d_raw_focus_16bits

############################ HHA ############################
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_HHA_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_HHA_8bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_HHA_focus_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_HHA_focus_8bits

############################ jet ############################
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_jet_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_jet_8bits
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_jet_focus_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_jet_focus_8bits

############################ cubehelix ############################
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_cubehelix_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_cubehelix_8bits
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_cubehelix_focus_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_cubehelix_focus_8bits

############################ histeq ############################
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_histeqRandom_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_histeqRandom_8bits
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_histeqBack_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_histeqBack_8bits
./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_histeqFront_8bits_nyud_v2_40gupta nyud_v2_test_d_raw_histeqFront_8bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_histeqRandom_16bits_nyud_v2_40gupta nyud_v2_test_d_raw_histeqRandom_16bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_histeqBack_16bits_nyud_v2_40gupta nyud_v2_test_d_raw_histeqBack_16bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_d_raw_histeqFront_16bits_nyud_v2_40gupta nyud_v2_test_d_raw_histeqFront_16bits

############################ rgb variable I ############################
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_rgb_i_10_8bits_nyud_v2_40gupta nyud_v2_test_rgb_i_10_8bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_rgb_i_20_8bits_nyud_v2_40gupta nyud_v2_test_rgb_i_20_8bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_rgb_i_30_8bits_nyud_v2_40gupta nyud_v2_test_rgb_i_30_8bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_rgb_i_40_8bits_nyud_v2_40gupta nyud_v2_test_rgb_i_40_8bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_rgb_i_50_8bits_nyud_v2_40gupta nyud_v2_test_rgb_i_50_8bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_rgb_i_60_8bits_nyud_v2_40gupta nyud_v2_test_rgb_i_60_8bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_rgb_i_70_8bits_nyud_v2_40gupta nyud_v2_test_rgb_i_70_8bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_rgb_i_80_8bits_nyud_v2_40gupta nyud_v2_test_rgb_i_80_8bits
#./experiments/scripts/jg_test_set_all_ao_nyud_v2_40gupta.sh ao_rgb_i_90_8bits_nyud_v2_40gupta nyud_v2_test_rgb_i_90_8bits

############################ rgb mixed ############################
#./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh ao_rgb_iAll_8bits_nyud_v2_40gupta nyud_v2_test_i_all_rgb_iAll_8bits
#./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh ao_rgb_iRange_100_60_8bits_nyud_v2_40gupta nyud_v2_test_i_100_60_rgb_iRange_100_60_8bits
#./experiments/scripts/train_rcnn_ao_nyud_v2_40gupta.sh ao_rgb_iRange_50_10_8bits_nyud_v2_40gupta nyud_v2_test_i_50_10_rgb_iRange_50_10_8bits
