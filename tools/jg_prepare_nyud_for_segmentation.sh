#/bin/bash

#jg_prepare_nyud_for_segmentation.sh /home/jguerry/workspace/datasets/NYUDV2

if [ "$#" -ne 1 ]; then
    echo "Usage: jg_prepare_nyud_for_segmentation.sh <path-to-NYUD>"
    echo "exemple: ./jg_prepare_nyud_for_segmentation.sh /c16/THESE.JORIS/datasets/NYUD_V2"
    echo "Illegal number of parameters"
    exit 1
fi

nyud_dir=$1
nyud_segmentation_dir=${nyud_dir}/data_segmentation
actual_path=pwd
#echo "Moving to ${nyud_dir}."
#cd ${nyud_dir}

for stage in 'trainGupta' 'valGupta' 'trainvalGupta' 'testGupta'
do
  echo "Processing ${stage} set :"
  for encoding in 'labels_gupta' 'd_raw_DNA_8bits' 'd_raw_focus_8bits' 'd_raw_focus_16bits' 'd_raw_normal_8bits' 'd_raw_normal_16bits' 'd_raw_HHA_8bits' 'd_raw_jet_8bits' 'd_raw_cubehelix_8bits' 'd_raw_histeqBack_8bits' 'd_raw_histeqBack_16bits' 'd_raw_histeqFront_8bits' 'd_raw_histeqFront_16bits' 'd_raw_histeqRandom_8bits' 'd_raw_histeqRandom_16bits' 'rgb_i_10_8bits' 'rgb_i_20_8bits' 'rgb_i_30_8bits' 'rgb_i_40_8bits' 'rgb_i_50_8bits' 'rgb_i_60_8bits' 'rgb_i_70_8bits' 'rgb_i_80_8bits' 'rgb_i_90_8bits' 'rgb_i_100_8bits'
    do
  	echo "Encoding :${encoding}"
      imageset_file=${nyud_dir}/data/sets/${stage}.txt
      image_dir=${nyud_dir}/data/${encoding}
      output_image_dir=${nyud_segmentation_dir}/${encoding}/${stage}
      mkdir -p ${output_image_dir}

      cd ${output_image_dir}
      while read -r file; do
          ln -sf "../../../data/${encoding}/${file}.png" .;
      done < ${imageset_file}
    done
done

cd ${actual_path}


echo "Done!"
