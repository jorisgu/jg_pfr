#!/bin/bash
###################################################################
###################################################################
############################# DEPTH ###############################
###################################################################
###################################################################

############################ DNA ############################
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_DNA_8bits nyud_v2_trainvalGupta_d_raw_DNA_8bits nyud_v2_testGupta_d_raw_DNA_8bits 20000

############################ HHA ############################
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_HHA_8bits nyud_v2_trainvalGupta_d_raw_HHA_8bits nyud_v2_testGupta_d_raw_HHA_8bits 20000

############################ depth raw ############################
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_normal_8bits nyud_v2_trainvalGupta_d_raw_normal_8bits nyud_v2_testGupta_d_raw_normal_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_normal_16bits nyud_v2_trainvalGupta_d_raw_normal_16bits nyud_v2_testGupta_d_raw_normal_16bits 20000

############################ histeq ############################
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_histeqRandom_8bits nyud_v2_trainvalGupta_d_raw_histeqRandom_8bits nyud_v2_testGupta_d_raw_histeqRandom_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_histeqBack_8bits nyud_v2_trainvalGupta_d_raw_histeqBack_8bits nyud_v2_testGupta_d_raw_histeqBack_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_histeqFront_8bits nyud_v2_trainvalGupta_d_raw_histeqFront_8bits nyud_v2_testGupta_d_raw_histeqFront_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_histeqRandom_16bits nyud_v2_trainvalGupta_d_raw_histeqRandom_16bits nyud_v2_testGupta_d_raw_histeqRandom_16bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_histeqBack_16bits nyud_v2_trainvalGupta_d_raw_histeqBack_16bits nyud_v2_testGupta_d_raw_histeqBack_16bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_histeqFront_16bits nyud_v2_trainvalGupta_d_raw_histeqFront_16bits nyud_v2_testGupta_d_raw_histeqFront_16bits 20000

############################ jet ############################
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_jet_8bits nyud_v2_trainvalGupta_d_raw_jet_8bits nyud_v2_testGupta_d_raw_jet_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_jet_focus_8bits nyud_v2_trainvalGupta_d_raw_jet_focus_8bits nyud_v2_testGupta_d_raw_jet_focus_8bits 20000

############################ cubehelix ############################
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_cubehelix_8bits nyud_v2_trainvalGupta_d_raw_cubehelix_8bits nyud_v2_testGupta_d_raw_cubehelix_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_cubehelix_focus_8bits nyud_v2_trainvalGupta_d_raw_cubehelix_focus_8bits nyud_v2_testGupta_d_raw_cubehelix_focus_8bits 20000

############################ focus 95% ############################
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_focus_8bits nyud_v2_trainvalGupta_d_raw_focus_8bits nyud_v2_testGupta_d_raw_focus_8bits 20000
./experiments/scripts/jg_basic_train_ao_nyudv2_vgg16.sh 0 ao_d_raw_focus_16bits nyud_v2_trainvalGupta_d_raw_focus_16bits nyud_v2_testGupta_d_raw_focus_16bits 20000
