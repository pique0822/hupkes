import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Explores dataset files and the distribution of numbers')
parser.add_argument('--file',type=str, default=None,
                    help='File location with which we will explore the results')
parser.add_argument('--data_type',type=str, default='implicit',
                    help='{implicit | explicit | repeated}')
parser.add_argument('--K',type=int, default=2,
                    help='Dataset K.')
parser.add_argument('--output_file', type=str, default='training_frequency.png',
					help='Name of the image that will be outputted.')
args = parser.parse_args()


all_chars = []
with open(args.file,'r') as dataset:
	for line in dataset:
		training_values = line.split(' ')[:args.K]
		all_chars.extend([int(x) for x in training_values])

counts = plt.hist(all_chars,bins=np.arange(11)-0.5, align='mid', edgecolor='black', linewidth=1.2)
plt.xticks(range(10),range(10))
plt.xlabel('Token')
plt.ylabel('Occurences')
plt.title('Occurences of Each Token in Dataset K='+str(args.K))
plt.savefig('L'+str(args.K)+'/'+args.output_file)

