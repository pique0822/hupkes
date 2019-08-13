from Gated_GRU import GatedGRU
from datasets.dataset import Dataset

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import seaborn as sns

import pandas as pd

import argparse

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description='Trains a GRU model on the arithmetic language dataset')
# model information
parser.add_argument('--embedding_size',type=int, default=2,
                    help='The dimension of the model embedding')
parser.add_argument('--hidden_size',type=int, default=15,
                    help='The dimension of the model hidden state')

# dataset_information
parser.add_argument('--vocabulary_file',type=str,
                    default='datasets/hupkes_vocabulary.txt',
                    help='File containing each possible word in the training set.')
parser.add_argument('--dataset_file',type=str, default='hupkes.txt',
                    help='File that contains the full dataset from which we will take the training and test data.')
parser.add_argument('--test_percent',type=float,
                    default=0.2,
                    help='Percent of data that is to be used as testing.')
# parser.add_argument('--model_save',type=str, default=None,
#                     help='File name to which we will save the model.')
args = parser.parse_args()

def apply(mode, value, symbol):
    word2val = ['zero','one','two','three','four','five','six','seven','eight','nine','ten']

    if symbol[0] == '-':
        symbol_value = -word2val.index(symbol[1:])
    else:
        symbol_value = word2val.index(symbol)

    if mode == 1: #addition
        return value + symbol_value
    elif mode == 0:
        return value - symbol_value

def cumulative_sum(expression):
    # cumulative strategy
    mode_stack = []
    # cumulative result
    result = 0
    # mode 1 is addition and mode 0 is subtraction
    mode = 1
    for symbol in expression:
        if symbol == '(':
            mode_stack.append(mode)
        elif symbol == ')':
            mode = mode_stack.pop()
        elif symbol == 'plus':
            pass
        elif symbol == 'minus':
            if mode == 0:
                mode = 1
            else:
                mode = 0
        else:
            pass
            result = apply(mode, result, symbol)
    return result

def recursive_sum(expression):
    result_stack = []
    mode_stack = []
    result = 0
    # mode 1 is addition and mode 0 is subtraction
    mode = 1
    for symbol in expression:
        if symbol == '(':
            mode_stack.append(mode)
            result_stack.append(result)
            result = 0
            mode = 1
        elif symbol == ')':
            mode = mode_stack.pop()
            prev_result = result_stack.pop()
            if mode == 1: # addition
                result = prev_result + result
            elif mode == 0:
                result = prev_result - result
        elif symbol == 'plus':
            mode = 1
        elif symbol == 'minus':
            mode = 0
        else:
            result = apply(mode, result, symbol)

    return result

def plot_model_MSE(dataset, show=False):
    mses_df = pd.DataFrame.from_dict(dataset)

    sns.pointplot(x='dataset_id',y='mse',data=mses_df)
    plt.title('Testing Model Performance')
    plt.xlabel('Dataset')
    plt.ylabel('MSE')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('results/model_perf_all.png')
        plt.close()


    sns.factorplot(x='dataset_id',y='mse',hue='model_id',data=mses_df)
    plt.title('Testing Model Performance by Model')
    plt.xlabel('Dataset')
    plt.ylabel('MSE')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('results/model_perf_by_model.png')
        plt.close()

def plot_decoder_MSE(dataset, show=False):
    decoder_df = pd.DataFrame.from_dict(dataset)

    sns.pointplot(x='dataset_id',y='mse', hue='target_type',data=decoder_df)
    plt.title('Testing Linear Decoder Performance')
    plt.xlabel('Dataset')
    plt.ylabel('MSE')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('results/linear_decoders.png')
        plt.close()

test_batch_size = 1

relevant_models = 20
relevant_datasets = 9
vocabulary = []
with open(args.vocabulary_file, 'r') as vocab_file:
    for line in vocab_file:
        vocabulary.append(line.strip())

# dataframe
# |  MODELID  |  DATASET  |  MSE  |

criterion = nn.MSELoss()

# MSE over all datasets models
MSE_per_model = {'model_id':[],'dataset_id':[],'mse':[]}

# mse over all datasets linear decoders
all_decoder_MSEs = {'model_id':[], 'dataset_id':[], 'target_type':[], 'mse':[]}

training_percent = 0.9

for k in range(relevant_models):
    print('Model '+str(k+1))

    model = GatedGRU(input_size = len(vocabulary),
                     embedding_size = args.embedding_size,
                     hidden_size = args.hidden_size,
                     output_size = 1)
    model.load_state_dict(torch.load('models/hupkes_model_'+str(k+1)+'.mdl'))
    model.eval()


    for dataset_number in range(relevant_datasets):
        print('L'+str(dataset_number+1))

        # loading dataset
        dataset = Dataset('datasets/L'+str(dataset_number+1)+'/'+args.dataset_file, 1 - args.test_percent, test_batch_size)
        dataset.set_vocabulary(vocabulary)

        cumulative_sum_dataset = []
        recursive_sum_dataset = []
        dataset_hiddens = []

        # getting data
        dataset_MSE = 0
        for data_idx in range(dataset.batched_testing_size()):
            X,y = dataset.get_datapoint(data_idx, training=False)

            for line in X:
                cumulative_sum_results = []
                recursive_sum_results = []
                words = []
                for token in line:
                    words.append(dataset.token2word(token))

                    cum_result = cumulative_sum(words)
                    cumulative_sum_results.append(cum_result)

                    rec_result = recursive_sum(words)
                    recursive_sum_results.append(rec_result)

                cumulative_sum_dataset.extend(cumulative_sum_results)
                recursive_sum_dataset.extend(recursive_sum_results)

            X = torch.Tensor(X).reshape(X.shape[0], -1, 1).long()
            y = torch.Tensor(y).reshape(X.shape[0],-1)



            hidden = model.init_hidden()

            output, hidden, (update_gates, reset_gates, hidden_states) = model(X,hidden)

            for t in range(len(hidden_states)):
                hidden_t = hidden_states[t].reshape(test_batch_size,-1)
                dataset_hiddens.append(hidden_t.detach().numpy())


            dataset_MSE += criterion(output,y).item()*X.shape[0]

        MSE_per_model['model_id'].append(k+1)
        MSE_per_model['dataset_id'].append('L'+str(dataset_number + 1))
        MSE_per_model['mse'].append(dataset_MSE/dataset.batched_testing_size())

        # linear decoders

        cum_sum = np.array(cumulative_sum_dataset)
        rec_sum = np.array(recursive_sum_dataset)

        hiddens = np.array(dataset_hiddens).reshape(len(cum_sum),-1)

        training_set = np.random.choice(range(len(hiddens)),len(hiddens),replace=True)

        testing_set = training_set[int(len(training_set)*training_percent):]
        training_set = training_set[:int(len(training_set)*training_percent)]

        # cumulative sum
        reg = LinearRegression()
        reg.fit(hiddens[training_set,:], cum_sum[training_set])

        predictions = reg.predict(hiddens[testing_set,:])

        mse = mean_squared_error(predictions, cum_sum[testing_set])

        all_decoder_MSEs['model_id'].append(k+1)
        all_decoder_MSEs['dataset_id'].append('L'+str(dataset_number+1))
        all_decoder_MSEs['mse'].append(mse)
        all_decoder_MSEs['target_type'].append('Cumulative Sum')

        # recursive sum
        reg = LinearRegression()
        reg.fit(hiddens[training_set,:], rec_sum[training_set])

        predictions = reg.predict(hiddens[testing_set,:])

        mse = mean_squared_error(predictions, rec_sum[testing_set])

        all_decoder_MSEs['model_id'].append(k+1)
        all_decoder_MSEs['dataset_id'].append('L'+str(dataset_number+1))
        all_decoder_MSEs['mse'].append(mse)
        all_decoder_MSEs['target_type'].append('Recursive Sum')

plot_model_MSE(MSE_per_model, show=False)
plot_decoder_MSE(all_decoder_MSEs, show=False)
