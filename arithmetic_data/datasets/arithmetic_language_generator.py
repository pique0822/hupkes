import argparse
import random

from number import *
# python3 arithmetic_language_generator.py --num_examples 10000 --k 9 --output_file L9/data.txt
parser = argparse.ArgumentParser(description='Generates a set of lines of arithmetic and the solution.')
parser.add_argument('--num_examples',type=int, default=100,
                    help='Integer that defines the number of examples sentences that should be generated using this PCFG.')
parser.add_argument('--k',type=int, default=1,
                    help='Quantity of numbers that are to be used in the arithmetic operation.')
parser.add_argument('--output_file',type=str, default='data.txt',
                    help='File that contains the generated lines from this program.')
parser.add_argument('--operation_type',type=str, default='addition',
                    help='{addition | all} This will determine what operations are within the range for arithmetic.')
args = parser.parse_args()

with open(args.output_file,'w+') as out:
    line = 0
    while line < args.num_examples:

        all_nums = []
        for seq in range(args.k):
            rand_int = random.randint(-10,10)
            rand_num = Number(rand_int)
            all_nums.append(rand_num)


        full_op = None
        for seq_num in all_nums:

            if args.operation_type == 'addition':
                if random.randint(0,1) == 0:
                    operation = '+'
                else:
                    operation = '-'
            else:
                op_idx = random.randint(0,3)
                if op_idx == 0:
                    operation = '+'
                elif op_idx == 1:
                    operation = '-'
                elif op_idx == 2:
                    operation = '*'
                else:
                    operation = '/'

            if random.randint(0,1) == 0:
                full_op = Operation(full_op, seq_num, operation)
            else:
                full_op = Operation(seq_num, full_op, operation)

        if full_op.get_value() == 'ERROR':
            continue
        else:
            line += 1

            out.write(str(full_op)+';'+str(full_op.get_value())+'\n')
