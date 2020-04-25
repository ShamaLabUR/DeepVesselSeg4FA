#!/bin/bash

# Root directory to dataset 
DATA_PATH='../datasets/single_sample_from_RECOVERY-FA19'

# Directory to pretrained model
PRETRAINED_DIR='../pretrained_models/'
PRETRAINED_ID=8

# Directory to results folder. Automatically generated if it doesn't exist
SAVE_DIR='../results'

# Batch size
BATCH_SIZE=16

python detect_FA_vessels_w_DNN.py -d ${DATA_PATH} -p ${PRETRAINED_DIR} -i ${PRETRAINED_ID} -s ${SAVE_DIR} -b ${BATCH_SIZE}
