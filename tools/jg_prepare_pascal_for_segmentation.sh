#/bin/bash

#jg_prepare_pascal_for_segmentation.sh /home/jguerry/workspace/datasets/PASCAL_VOC 2007 /home/jguerry/workspace/datasets/PASCAL_VOC_segmentation/data

if [ "$#" -ne 3 ]; then
    echo "Usage: jg_prepare_pascal_for_segmentation.sh <path-to-PASCAL-VOC> <year> <output_dir>"
    echo "exemple: jg_prepare_pascal_for_segmentation.sh /home/jguerry/workspace/datasets/PASCAL_VOC 2007 /home/jguerry/workspace/datasets/PASCAL_VOC_segmentation"
    echo "Illegal number of parameters"
    exit 1
fi

pascal_path=$1
year=$2
output_dir=$3


pascal_dir=${pascal_path}/VOCdevkit/VOC${year}

for stage in 'train' 'val' 'trainval' #'test'
do
	echo "Processing ${stage} data"
    imageset_file=${pascal_dir}/ImageSets/Segmentation/${stage}.txt
    image_dir=${pascal_dir}/JPEGImages
    label_dir=${pascal_dir}/SegmentationClass

    output_image_dir=${output_dir}/data/VOC${year}/${stage}/images/
    output_label_dir=${output_dir}/data/VOC${year}/${stage}/labels/

    mkdir -p ${output_image_dir}
    mkdir -p ${output_label_dir}

    while read -r file; do
        cp "${image_dir}/${file}.jpg" ${output_image_dir};
        cp "${label_dir}/${file}.png" ${output_label_dir};
    done < ${imageset_file}
done

echo "Done!"
