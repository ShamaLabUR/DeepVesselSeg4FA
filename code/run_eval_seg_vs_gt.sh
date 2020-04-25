#!/bin/bash

# By default the code performs evaluation on the single_sample_from_RECOVERY-FA19 dataset
# Root directory to dataset 
DATA_PATH='../data/datasets/single_sample_from_RECOVERY-FA19'
# Directory to results
SAVE_DIR='../results'


# If you download the full RECOVERY-FA19 dataset and place them in the datasets folder, you can run the evaluation for all images from the RECOVERY-FA19 dataset. To do so, uncomment the next two lines to set the DATA_PATH and SAVE_DIR 
#DATA_PATH='../data/datasets/RECOVERY-FA19'
#SAVE_DIR='../data/pretrained_results'

python eval_seg_vs_gt.py -d ${DATA_PATH}  -s ${SAVE_DIR} 
