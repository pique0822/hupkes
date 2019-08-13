#!/bin/bash



SINGULARITY_IMG=/om2/user/jgauthie/singularity_images/deepo-cpu.simg
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd -P )"

MODEL_ID=$1
NUM_EPOCHS=$2

singularity exec -B /om/group/cpl -B "$PROJECT_PATH" "$SINGULARITY_IMG" python3 train_GRU.py \
		--model_save "models/hupkes_model_${MODEL_ID}.mdl" \
		--num_epochs ${NUM_EPOCHS} > "models/training_model_${MODEL_ID}.txt"
