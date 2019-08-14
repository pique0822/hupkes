#!/bin/bash

DATASET_NAME=ten_tokens
VOCAB_FILE=${DATASET_NAME}.txt
OPERATION=combined


for k in $(seq 2 10)
do
	if [ ! -d "L${k}" ]
	then
		mkdir L${k}
	fi

	echo Generating L${k}
	python3 sequence_generator.py --num_examples 5000 --k ${k} --operation_type ${OPERATION} --output_file L${k}/${DATASET_NAME}_${OPERATION}_data.txt
done
