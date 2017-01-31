#/bin/bash

#jg_prepare_nyud_for_segmentation.sh /home/jguerry/workspace/datasets/NYUDV2

if [ "$#" -ne 3 ]; then
    echo "Usage: jg_extract_set_from_dataset.sh <path-to-origin-dataset> <path-to-output-dataset> <path-to-subsets"
    echo "exemple: ./jg_extract_set_from_dataset.sh /c16/THESE.JORIS/datasets/SUNRGBD_pvf/ data_nyudv2 nyudv2_gupta"
    echo "Illegal number of parameters"
    exit 1
fi

input_dataset_dir=$1/data/
output_dataset_dir=$1/$2/
sets_dir=$1/data/sets/$3/




for stage in 'trainGupta' 'valGupta' 'trainvalGupta' 'testGupta'
do
  echo "Processing ${stage} set :"
  for encoding in 'labels_segmentation_37' 'd_raw_cubehelix_8bits' 'd_raw_histeqBack_8bits' 'd_raw_histeqBack_16bits' 'd_raw_histeqBack_cubehelix_8bits' 'd_raw_histeqBack_jet_8bits' 'd_raw_histeqFront_8bits' 'd_raw_histeqFront_16bits' 'd_raw_histeqFront_cubehelix_8bits' 'd_raw_histeqFront_jet_8bits' 'd_raw_histeqRandom_8bits' 'd_raw_histeqRandom_16bits' 'd_raw_histeqRandom_cubehelix_8bits' 'd_raw_histeqRandom_jet_8bits' 'd_raw_jet_8bits' 'd_raw_normal_8bits' 'd_raw_normal_16bits' 'rgbd_egh_8bits' 'rgb_i_10_8bits' 'rgb_i_20_8bits' 'rgb_i_30_8bits' 'rgb_i_40_8bits' 'rgb_i_50_8bits' 'rgb_i_60_8bits' 'rgb_i_70_8bits' 'rgb_i_80_8bits' 'rgb_i_90_8bits' 'rgb_i_100_8bits'
    do
  	echo "Encoding : ${encoding}"
      imageset_file=${sets_dir}${stage}.txt
      image_dir=${input_dataset_dir}${encoding}/
      output_image_dir=${output_dataset_dir}${encoding}/${stage}/
      mkdir -p ${output_image_dir}

      cd ${output_image_dir}
      while read -r file; do
          ln -sf "../../../data/${encoding}/${file}.png" .;
      done < ${imageset_file}
    done
done

echo "Done!"
