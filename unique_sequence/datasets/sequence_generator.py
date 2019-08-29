import argparse
import random

import numpy as np


def generate_examples(transition_type, operation_type, vocabulary, k, num_examples):

    transition_token = ' '
    transition_separation = 0
    if transition_type == 'explicit':
        transition_token = ' ' + vocabulary[len(vocabulary) - 1] + ' '
        transition_separation = 1

    output = ""
    line = 0
    while line < num_examples:

        # import pdb; pdb.set_trace()
        # 1. Pick 'k' items from the vocabulary, call them 'unique'
        unique = np.random.choice(vocabulary[:len(vocabulary) - transition_separation],k,replace=False)

        # import pdb; pdb.set_trace()
        # 2. Pick 'k-1' items from the chosen 'unique' set, call them 'repeated'
        repeated = np.random.choice(unique,k,replace=False)

        if operation_type == 'singular':
            num_outputs = 1
        elif operation_type == 'combined':
            num_outputs = np.random.randint(1,k)
        else:
            num_outputs = 1

        single = repeated[k-num_outputs:]
        repeated = repeated[:k-num_outputs]

        if transition_type == 'repeated':
            if unique[len(unique)-1] == repeated[0]:
                if k == 2:
                    continue
                else:
                    temp = repeated[1]
                    repeated[1] = repeated[0]
                    repeated[0] = temp

            transition_token = ' '+repeated[0]+' '

        # import pdb; pdb.set_trace()
        # 3. Write the 'unique' set followed by the 'repeated' set. Finally write the item that is not repeated.

        example = ' '.join(unique) + transition_token+' '.join(repeated) + ';'+' '.join(single)
            
        output += example+'\n'
        line += 1
    return output

if __name__ == '__main__':
    main()

def main():
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
                        help='{implicit | explicit | repeated} This will determine if there is an explicit token encoding the beginning of the forgetting stage.')

    # dataset information
    parser.add_argument('--vocabulary_file',type=str, default='ten_tokens.txt',
                        help='File containing all possible tokens that can be used in this model. By default, the final token will be used as the explicit separation if set by the previous flag.')
    args = parser.parse_args()


    vocabulary = []
    with open(args.vocabulary_file, 'r') as vocab_file:
        for line in vocab_file:
            vocabulary.append(line.strip())

    output = generate_examples(args.transition_type, args.operation_type, vocabulary, args.k, args.num_examples)
    with open(args.output_file,'w+') as out:

        # import pdb; pdb.set_trace()
        out.write(output+'\n')
