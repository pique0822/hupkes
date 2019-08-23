import argparse

import matplotlib.pyplot as plt
import numpy as np

for K in range(2,11):
	print('\n\nL'+str(K))
	file = 'L'+str(K)+'/ten_tokens_repeated_singularnotriples_data.txt'

	repeated_examples = []
	nonrepeat_examples= []
	with open(file,'r') as dataset:
		for line in dataset:
			triple_ocurrence = False
			training_values = line.split(';')[0].split(' ')
			char_2 = ""
			char_1 = ""
			char_0 = ""
			for char in training_values:
				char_0 = char

				if char_0 == char_1 and char_1 == char_2:
					triple_ocurrence = True

				char_2 = char_1
				char_1 = char_0
				

			if triple_ocurrence:
				repeated_examples.append(line)
			else:
				nonrepeat_examples.append(line)

				


	print('Occurences of Triple Repetition: ',len(repeated_examples))
	print('Occurences of Double Repetition: ',len(nonrepeat_examples))


