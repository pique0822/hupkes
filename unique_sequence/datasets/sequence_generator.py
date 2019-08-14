import argparse
import random

import numpy as np

# python3 arithmetic_language_generator.py --num_examples 10000 --k 9 --output_file L9/data.txt
parser = argparse.ArgumentParser(description='Generates a set of sequences generated to retain information.')
parser.add_argument('--num_examples',type=int, default=100,
                    help='Integer that defines the number of examples sentences that should be generated.')
parser.add_argument('--k',type=int, default=2,
                    help='Quantity of tokens that are to be used as possible memories.')
parser.add_argument('--output_file',type=str, default='data.txt',
                    help='File that contains the generated lines from this program.')
parser.add_argument('--operation_type',type=str, default='singular',
                    help='{singular | combined} This will determine if one or multiple tokens are expected as output')
parser.add_argument('--transition_type',type=str, default='implicit',
                    help='{implicit | explicit} This will determine if there is an explicit token encoding the beginning of the forgetting stage.')

# dataset information
parser.add_argument('--vocabulary_file',type=str, default='ten_tokens.txt',
                    help='File containing all possible tokens that can be used in this model. By default, the final token will be used as the explicit separation if set by the previous flag.')
args = parser.parse_args()


vocabulary = []
with open(args.vocabulary_file, 'r') as vocab_file:
    for line in vocab_file:
        vocabulary.append(line.strip())

transition_token = ' '
transition_separation = 0
if args.transition_type == 'explicit':
    transition_token = ' ' + vocabulary[len(vocabulary) - 1] + ' '
    transition_separation = 1

with open(args.output_file,'w+') as out:
    line = 0
    while line < args.num_examples:

        # import pdb; pdb.set_trace()
        # 1. Pick 'k' items from the vocabulary, call them 'unique'
        unique = np.random.choice(vocabulary[:len(vocabulary) - transition_separation],args.k,replace=False)

        # import pdb; pdb.set_trace()
        # 2. Pick 'k-1' items from the chosen 'unique' set, call them 'repeated'
        repeated = np.random.choice(unique,args.k,replace=False)

        if args.operation_type == 'singular':
            num_outputs = 1
        elif args.operation_type == 'combined':
            num_outputs = np.random.randint(1,args.k)
        else:
            num_outputs = 1

        single = repeated[args.k-num_outputs:]
        repeated = repeated[:args.k-num_outputs]

        # import pdb; pdb.set_trace()
        # 3. Write the 'unique' set followed by the 'repeated' set. Finally write the item that is not repeated.

        example = ' '.join(unique) + transition_token+' '.join(repeated) + ';'+' '.join(single)

        # import pdb; pdb.set_trace()
        out.write(example+'\n')
        line += 1
