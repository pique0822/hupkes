#!/bin/bash

> training_ids.txt

NUM_EPOCHS=1000
for MODEL_ID in $(seq 1 20)
do
	echo Training Model ${MODEL_ID} for ${NUM_EPOCHS} epochs

	sbatch --time 1000 --job-name hupkes_${MODEL_ID} Lk_training.sh ${MODEL_ID} ${NUM_EPOCHS} >> training_ids.txt
done
