#!/bin/bash


SINGULARITY_IMG=/om2/user/jgauthie/singularity_images/deepo-cpu.simg
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd -P )"


singularity exec -B /om/group/cpl -B "$PROJECT_PATH" "$SINGULARITY_IMG" python3 MSE_plotting.py
