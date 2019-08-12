#!/bin/bash

NUM_EPOCHS=1000
for MODEL_ID in $(seq 1 20)
do
	echo Training Model ${MODEL_ID} for ${NUM_EPOCHS} epochs

	sbatch --time 5 --job-name hupkes_${MODEL_ID} --gres=gpu:tesla-k80:1 Lk_training.sh ${MODEL_ID} ${NUM_EPOCHS}
done
