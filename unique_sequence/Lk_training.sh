#!/bin/bash



SINGULARITY_IMG=/om2/user/jgauthie/singularity_images/deepo-cpu.simg
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd -P )"

BASE_NAME=$1
MODEL_ID=$2
NUM_EPOCHS=$3
DATASET=$4
VOCAB_FILE=$5

singularity exec -B /om/group/cpl -B "$PROJECT_PATH" "$SINGULARITY_IMG" python3 train_GRU.py \
		--model_save "models/${BASE_NAME}_${MODEL_ID}.mdl" \
		--num_epochs ${NUM_EPOCHS} > "models/training_${BASE_NAME}_${MODEL_ID}.txt"\
		--dataset_file ${DATASET} \
		--vocabulary_file ${VOCAB_FILE}
