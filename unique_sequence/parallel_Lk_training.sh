#!/bin/bash

TIME=3000
NUM_EPOCHS=3000
DATASET=ten_tokens_explicit_singular_data.txt
BASE_NAME=ten_tokens_seeded_explicit
VOCAB_FILE=datasets/ten_tokens_explicit.txt

> ${BASE_NAME}_training_ids.txt


for MODEL_ID in $(seq 1 20)
do
	echo Training Model ${BASE_NAME}_${MODEL_ID} for ${NUM_EPOCHS} epochs

	sbatch --time ${TIME} --job-name ${BASE_NAME}_${MODEL_ID} Lk_training.sh ${BASE_NAME} ${MODEL_ID} ${NUM_EPOCHS} ${DATASET} ${VOCAB_FILE} >> ${BASE_NAME}_training_ids.txt
done
