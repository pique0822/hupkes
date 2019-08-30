#!/bin/bash

TIME=3000
NUM_EPOCHS=1000
DATASET=ten_tokens_explicit_singular_data.txt
BASE_NAME=transformer
VOCAB_FILE=datasets/ten_tokens_explicit.txt

> ${BASE_NAME}_training_ids.txt


for HIDDEN_SIZE in $(seq 16 32)
do
	for NUM_LAYERS in $(seq 2 6)
	do
		for NUM_HEADS in $(seq 2 8)
		do 
			echo Training Model ${BASE_NAME}_${MODEL_ID} for ${NUM_EPOCHS} epochs

			sbatch --time ${TIME} --job-name ${BASE_NAME}_${MODEL_ID} transformer_training.sh ${BASE_NAME} ${MODEL_ID} ${NUM_EPOCHS} ${DATASET} ${VOCAB_FILE} ${NUM_LAYERS} ${NUM_HEADS} ${HIDDEN_SIZE} >> ${BASE_NAME}_training_ids.txt
		done
	done
done
