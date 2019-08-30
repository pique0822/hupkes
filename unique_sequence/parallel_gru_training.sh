#!/bin/bash

TIME=3000
NUM_EPOCHS=5000
DATASET=alphabet_explicit_singular_data.txt
BASE_NAME=alphabet_explicit
VOCAB_FILE=datasets/alphabet_explicit.txt

> ${BASE_NAME}_training_ids.txt


for MODEL_ID in $(seq 1 20)
do
	echo Training Model ${BASE_NAME}_${MODEL_ID} for ${NUM_EPOCHS} epochs

	sbatch --time ${TIME} --job-name ${BASE_NAME}_${MODEL_ID} gru_training.sh ${BASE_NAME} ${MODEL_ID} ${NUM_EPOCHS} ${DATASET} ${VOCAB_FILE} >> ${BASE_NAME}_training_ids.txt
done
