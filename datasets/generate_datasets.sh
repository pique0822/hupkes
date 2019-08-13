#!/bin/bash

for k in $(seq 1 9)
do
	echo Generating L${k}
	python3 arithmetic_language_generator.py --num_examples 5000 --k ${k} --operation_type all --output_file L${k}/all_operations.txt
done
