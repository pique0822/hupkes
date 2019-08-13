import argparse
import random

import numpy as np

# python3 arithmetic_language_generator.py --num_examples 10000 --k 9 --output_file L9/data.txt
parser = argparse.ArgumentParser(description='Generates a set of sequences generated to retain information.')
parser.add_argument('--num_examples',type=int, default=100,
                    help='Integer that defines the number of examples sentences that should be generated.')
parser.add_argument('--k',type=int, default=1,
                    help='Quantity of tokens that are to be used as possible memories.')
parser.add_argument('--output_file',type=str, default='data.txt',
                    help='File that contains the generated lines from this program.')
parser.add_argument('--operation_type',type=str, default='singular',
                    help='{singular | combined} This will determine if one or multiple tokens are expected as output')

# dataset information
parser.add_argument('--vocabulary_file',type=str, default='ten_tokens.txt',
                    help='File containing all possible tokens that can be used in this model.')
args = parser.parse_args()


vocabulary = []
with open(args.vocabulary_file, 'r') as vocab_file:
    for line in vocab_file:
        vocabulary.append(line.strip())


with open(args.output_file,'w+') as out:
    line = 0
    while line < args.num_examples:

        # 1. Pick 'k' items from the vocabulary, call them 'unique'
        unique = np.random.choice(vocabulary,k,replace=False)
        import pdb; pdb.set_trace()
        # 2. Pick 'k-1' items from the chosen 'unique' set, call them 'repeated'

        # 3. Write the 'unique' set followed by the 'repeated' set. Finally write the item that is not repeated.


        line += 1

        out.write(str(full_op)+';'+str(full_op.get_value())+'\n')
