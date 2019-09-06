#!/bin/bash


SINGULARITY_IMG=/om2/user/jgauthie/singularity_images/deepo-cpu.simg
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd -P )"

BASE_NAME=$1
MODEL_ID=$2
NUM_EPOCHS=$3
DATASET=$4
VOCAB_FILE=$5

NUM_LAYERS=$6
NUM_HEADS=$7
HIDDEN_SIZE=$8

singularity exec -B /om/group/cpl -B "$PROJECT_PATH" "$SINGULARITY_IMG" python3 train_transformer.py \
		--model_save "${BASE_NAME}_${MODEL_ID}_H${NUM_HEADS}_L${NUM_LAYERS}_S{HIDDEN_SIZE}.mdl" \
		--num_epochs ${NUM_EPOCHS} \
		--num_layers ${NUM_LAYERS} \
		--num_heads ${NUM_HEADS} \
		--hidden_size ${HIDDEN_SIZE} \
		--dataset_file ${DATASET} \
		--vocabulary_file ${VOCAB_FILE} \
		--dataset_seed ${MODEL_ID} > "training_transformer_${BASE_NAME}_${MODEL_ID}.txt"
