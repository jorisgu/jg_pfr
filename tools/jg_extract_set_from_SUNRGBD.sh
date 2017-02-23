#/bin/bash

#jg_prepare_nyud_for_segmentation.sh /home/jguerry/workspace/datasets/NYUDV2

if [ "$#" -ne 3 ]; then
    echo "Usage: jg_extract_set_from_SUNRGBD.sh <path-to-origin-dataset> <path-to-output-dataset> <path-to-subsets"
    echo "exemple: ./jg_extract_set_from_SUNRGBD.sh /home/jguerry/workspace/datasets/SUNRGBD_pv data_sunrgbd sunrgbd"
    echo "Illegal number of parameters"
    exit 1
fi

input_dataset_dir=$1
output_dataset_dir=$1/$2/
sets_dir=$1/data/sets/$3/



for stage in 'train' 'test'
do
  echo "Processing ${stage} set :"
  for encoding in 'rgb_i_100_8bits' #'labels_segmentation_37d'
    do
  	  echo "Encoding : ${encoding}"
      imageset_file=${sets_dir}${stage}.txt
      echo "set file: ${imageset_file}"
      output_image_dir=${output_dataset_dir}${encoding}/${stage}/
      echo "output_image_dir: ${output_image_dir}"
      mkdir -p ${output_image_dir}

      cd ${output_image_dir}
      while read -r file; do
          ln -sf "../../../data/${encoding}/${file}.png" .;
          #ln -sf "../../../data_spreadout/${encoding}/${file}.png" .;
      done < ${imageset_file}
    done
done
echo "Done!"
#
#
#
# for stage in 'train' 'test'
# do
#   echo "Processing ${stage} set :"
#   for encoding in 'd_raw_HHA_8bits' 'd_raw_DHA_8bits'
#     do
#   	  echo "Encoding : ${encoding}"
#       imageset_file=${sets_dir}${stage}.txt
#       echo "set file: ${imageset_file}"
#       output_image_dir=${output_dataset_dir}${encoding}/${stage}/
#       echo "output_image_dir: ${output_image_dir}"
#       mkdir -p ${output_image_dir}
#
#       cd ${output_image_dir}
#       while read -r file; do
#           ln -sf "../../../data/${encoding}/${file}.png" .;
#           #ln -sf "../../../data_spreadout/${encoding}/${file}.png" .;
#       done < ${imageset_file}
#     done
# done
#
# echo "Done!"
#
#
# for stage in 'train' 'test'
# do
#   echo "Processing ${stage} set :"
#   for encoding in 'd_raw_cubehelix_8bits' 'd_raw_jet_8bits'
#     do
#   	echo "Encoding : ${encoding}"
#       imageset_file=${sets_dir}${stage}.txt
#       output_image_dir=${output_dataset_dir}${encoding}_norm/${stage}/
#       mkdir -p ${output_image_dir}
#
#       cd ${output_image_dir}
#       while read -r file; do
#           #ln -sf "../../../data/${encoding}/${file}.png" .;
#           ln -sf "../../../data_spreadout/${encoding}/${file}.png" .;
#       done < ${imageset_file}
#     done
# done
#
# echo "Done!"
