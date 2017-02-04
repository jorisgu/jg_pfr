#/bin/bash

#jg_prepare_nyud_for_segmentation.sh /home/jguerry/workspace/datasets/NYUDV2

if [ "$#" -ne 3 ]; then
    echo "Usage: jg_extract_set_from_dataset.sh <path-to-origin-dataset> <path-to-output-dataset> <path-to-subsets"
    echo "exemple: ./jg_extract_set_from_dataset.sh /home/jogue/workspace/datasets/SUNRGBD_pv/ data_nyudv2 nyudv2_gupta"
    echo "Illegal number of parameters"
    exit 1
fi

input_dataset_dir=$1/data/
output_dataset_dir=$1/$2/
sets_dir=$1/data/sets/$3/




for stage in 'trainvalGupta' 'testGupta' 'trainGupta' 'valGupta'
do
  echo "Processing ${stage} set :"
  #for luminosity in '10' '20' '30' '40' '50' '60' '70' '80' '90' '100'
  #for luminosity in '60' '70' '80' '90' '100'
  for luminosity in '80' '90' '100'
    do
  	echo "Luminosity : ${luminosity}"
      imageset_file=${sets_dir}${stage}.txt
      output_image_dir=${output_dataset_dir}rgb_i_range_80-10-100/${stage}/
      output_label_dir=${output_dataset_dir}labels_segmentation_37d_range_80-10-100/${stage}/
      mkdir -p ${output_image_dir}
      mkdir -p ${output_label_dir}


      while read -r file; do
          cd ${output_image_dir}
          ln -sf "../../../data/rgb_i_${luminosity}_8bits/${file}.png" "./${file}_${luminosity}.png";
          cd ${output_label_dir}
          ln -sf "../../../data/labels_segmentation_37d/${file}.png" "./${file}_${luminosity}.png";
      done < ${imageset_file}
    done
done

echo "Done!"
