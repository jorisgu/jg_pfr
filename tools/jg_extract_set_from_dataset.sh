#/bin/bash

#jg_prepare_nyud_for_segmentation.sh /home/jguerry/workspace/datasets/NYUDV2

if [ "$#" -ne 3 ]; then
    echo "Usage: jg_extract_set_from_dataset.sh <path-to-origin-dataset> <path-to-output-dataset> <path-to-subsets"
    echo "exemple: ./jg_extract_set_from_dataset.sh /home/jogue/workspace/datasets/SUNRGBD_pv/ data_nyudv2 nyudv2_gupta"
    echo "Illegal number of parameters"
    exit 1
fi

input_dataset_dir=$1/data/
#input_dataset_dir=$1/data_spreadout/
output_dataset_dir=$1/$2/
sets_dir=$1/data/sets/$3/




for stage in 'trainvalGupta' 'testGupta' 'trainGupta' 'valGupta'
do
  echo "Processing ${stage} set :"
  for encoding in 'd_raw_DEA_8bits' 'd_raw_DHA_8bits' 'd_raw_HES_8bits' 'd_raw_PLDI_8bits'
    do
  	echo "Encoding : ${encoding}"
      imageset_file=${sets_dir}${stage}.txt
      output_image_dir=${output_dataset_dir}${encoding}/${stage}/
      mkdir -p ${output_image_dir}

      cd ${output_image_dir}
      while read -r file; do
          ln -sf "../../../data/${encoding}/${file}.png" .;
          #ln -sf "../../../data_spreadout/${encoding}/${file}.png" .;
      done < ${imageset_file}
    done
done

echo "Done!"
