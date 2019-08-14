#!/bin/bash

NUM_EPOCHS=3000
DATASET=all_operations.txt
BASE_NAME=all_operations
VOCAB_FILE=datasets/all_operations_vocabulary.txt
LEARNING_RATE=0.01

> ${BASE_NAME}_training_ids.txt


for MODEL_ID in $(seq 1 20)
do
	echo Training Model ${BASE_NAME}_${MODEL_ID} for ${NUM_EPOCHS} epochs

	sbatch --time 5000 --job-name ${BASE_NAME}_${MODEL_ID} Lk_training.sh ${BASE_NAME} ${MODEL_ID} ${NUM_EPOCHS} ${DATASET} ${VOCAB_FILE} ${LEARNING_RATE} >> ${BASE_NAME}_training_ids.txt
done
