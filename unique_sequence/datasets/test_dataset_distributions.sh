#!/bin/bash

for k in $(seq 2 10)
do
	echo Dataset L${k}
	python3 dataset_histograms.py \
	 --file L${k}/ten_tokens_explicit_singular_data.txt \
	 --K ${k} \
	 --data_type explicit \
	 --output_file training_distribution_explicit.png


	python3 dataset_histograms.py \
	 --file L${k}/ten_tokens_singular_data.txt \
	 --K ${k} \
	 --data_type implicit \
	 --output_file training_distribution_implicit.png

	python3 dataset_histograms.py \
	 --file L${k}/ten_tokens_repeated_singular_data.txt \
	 --K ${k} \
	 --data_type repeated \
	 --output_file training_distribution_repeated.png
done