#!/bin/bash

DATASET_NAME=alphabet_explicit
VOCAB_FILE=alphabet_explicit.txt
OPERATION=singular
TRANSITION=explicit


for k in $(seq 2 10)
do
	if [ ! -d "L${k}" ]
	then
		mkdir L${k}
	fi

	echo Generating L${k}
	python3 sequence_generator.py \
	--num_examples 5000 \
	--k ${k} \
	--operation_type ${OPERATION} \
	--transition_type ${TRANSITION} \
	--vocabulary_file ${VOCAB_FILE} \
	--output_file L${k}/${DATASET_NAME}_${OPERATION}_data.txt
done
