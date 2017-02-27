#/bin/bash

#jg_prepare_nyud_for_segmentation.sh /home/jguerry/workspace/datasets/NYUDV2

if [ "$#" -ne 3 ]; then
    echo "Usage: jg_extract_set_from_SUNRGBD.sh <path-to-origin-dataset> <path-to-output-dataset> <path-to-subsets"
    echo "exemple: ./jg_extract_set_from_InOut.sh /home/jogue/workspace/datasets/InOutDoorPeople data_inout ImageSets"
    echo "Illegal number of parameters"
    exit 1
fi

input_dataset_dir=$1
output_dataset_dir=$1/$2/
sets_dir=$1/$3/



for stage in 'trainval' 'test'
do
  echo "Processing ${stage} set :"
  for encoding in 'Images' 'Depth' 'labels_mask_person'
    do
  	echo "Encoding : ${encoding}"
      imageset_file=${sets_dir}${stage}.txt
      output_image_dir=${output_dataset_dir}${encoding}/${stage}/
      mkdir -p ${output_image_dir}

      cd ${output_image_dir}
      while read -r file; do
          cp -sf "../../../${encoding}/${file}.png" .;
      done < ${imageset_file}
    done
done

echo "Done!"
