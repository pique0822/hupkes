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

test_batch_size = 1

# for now we ignore the dataset file
L1 = Dataset('datasets/L1/data.txt', 0, test_batch_size)
L2 = Dataset('datasets/L2/data.txt', 0, test_batch_size)
L3 = Dataset('datasets/L3/data.txt', 0, test_batch_size)
L4 = Dataset('datasets/L4/data.txt', 0, test_batch_size)
L5 = Dataset('datasets/L5/data.txt', 0, test_batch_size)
L6 = Dataset('datasets/L6/data.txt', 0, test_batch_size)
L7 = Dataset('datasets/L7/data.txt', 0, test_batch_size)
L8 = Dataset('datasets/L6/data.txt', 0, test_batch_size)
L9 = Dataset('datasets/L7/data.txt', 0, test_batch_size)

vocabulary = ['zero', 'one', 'two', 'three', 'four', 'five',
              'six', 'seven', 'eight', 'nine', 'ten',
              '-one',  '-two', '-three', '-four', '-five',
              '-six', '-seven', '-eight', '-nine', '-ten',
              '(', ')', 'plus', 'minus']

L1.set_vocabulary(vocabulary)
L2.set_vocabulary(vocabulary)
L3.set_vocabulary(vocabulary)
L4.set_vocabulary(vocabulary)
L5.set_vocabulary(vocabulary)
L6.set_vocabulary(vocabulary)
L7.set_vocabulary(vocabulary)
L8.set_vocabulary(vocabulary)

# dataframe
# |  MODELID  |  DATASET  |  MSE  |


datasets = [L1,L2,L3,L4,L5,L6,L7,L8,L9]
models = []
for k in range(20):
    model = GatedGRU(input_size = len(vocabulary),
                     embedding_size = args.embedding_size,
                     hidden_size = args.hidden_size,
                     output_size = 1)
    model.load_state_dict(torch.load('models/hupkes_model_'+str(k+1)+'.mdl'))
    model.eval()
    models.append(model)

criterion = nn.MSELoss()

# MSE over all datasets
MSE_per_model = {'model_id':[],'dataset_id':[],'mse':[]}
hiddens_per_model = {}

cumulative_sum_dataset = {}
recursive_sum_dataset = {}
for k, model in enumerate(models):
    dataset_hiddens = {}
    print('Model '+str(k+1))
    for dataset_number, dataset in enumerate(datasets):
        cumulative_sum_dataset[dataset_number] = []
        recursive_sum_dataset[dataset_number] = []
        dataset_hiddens[dataset_number] = []

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

                cumulative_sum_dataset[dataset_number].extend(cumulative_sum_results)
                recursive_sum_dataset[dataset_number].extend(recursive_sum_results)

            X = torch.Tensor(X).reshape(X.shape[0], -1, 1).long()
            y = torch.Tensor(y).reshape(X.shape[0],-1)



            hidden = model.init_hidden()

            output, hidden, (update_gates, reset_gates, hidden_states) = model(X,hidden)

            for t in range(len(hidden_states)):
                hidden_t = hidden_states[t].reshape(test_batch_size,-1)
                dataset_hiddens[dataset_number].append(hidden_t.detach().numpy())


            dataset_MSE += criterion(output,y).item()*X.shape[0]

        MSE_per_model['model_id'].append(k+1)
        MSE_per_model['dataset_id'].append('L'+str(dataset_number + 1))
        MSE_per_model['mse'].append(dataset_MSE/dataset.batched_testing_size())

    hiddens_per_model[k+1] = dataset_hiddens



mses_df = pd.DataFrame.from_dict(MSE_per_model)

sns.pointplot(x='dataset_id',y='mse',data=mses_df)
plt.title('Testing Model Performance')
plt.xlabel('Dataset')
plt.ylabel('MSE')
plt.tight_layout()
plt.savefig('results/model_perf_all.png')
plt.close()


sns.factorplot(x='dataset_id',y='mse',hue='model_id',data=mses_df)
plt.title('Testing Model Performance by Model')
plt.xlabel('Dataset')
plt.ylabel('MSE')
plt.tight_layout()
plt.savefig('results/model_perf_by_model.png')
plt.close()



# Predicting cummulative sum
# training percentage for linear decoders

all_decoder_MSEs = {'model_id':[], 'dataset_id':[], 'target_type':[], 'mse':[]}
for model_id in hiddens_per_model.keys():
    decoder_mse = []
    training_percent = 0.9
    for dataset_number, dataset in enumerate(datasets):
        cum_sum = np.array(cumulative_sum_dataset[dataset_number])
        rec_sum = np.array(recursive_sum_dataset[dataset_number])

        hiddens = np.array(hiddens_per_model[model_id][dataset_number]).reshape(len(cum_sum),-1)

        training_set = np.random.choice(range(len(hiddens)),len(hiddens),replace=True)

        testing_set = training_set[int(len(training_set)*training_percent):]
        training_set = training_set[:int(len(training_set)*training_percent)]

        reg = LinearRegression()
        reg.fit(hiddens[training_set,:], cum_sum[training_set])

        predictions = reg.predict(hiddens[testing_set,:])

        mse = mean_squared_error(predictions, cum_sum[testing_set])

        all_decoder_MSEs['model_id'].append(model_id+1)
        all_decoder_MSEs['dataset_id'].append('L'+str(dataset_number+1))
        all_decoder_MSEs['mse'].append(mse)
        all_decoder_MSEs['target_type'].append('Cumulative Sum')


        reg = LinearRegression()
        reg.fit(hiddens[training_set,:], rec_sum[training_set])

        predictions = reg.predict(hiddens[testing_set,:])

        mse = mean_squared_error(predictions, rec_sum[testing_set])

        all_decoder_MSEs['model_id'].append(model_id+1)
        all_decoder_MSEs['dataset_id'].append('L'+str(dataset_number+1))
        all_decoder_MSEs['mse'].append(mse)
        all_decoder_MSEs['target_type'].append('Recursive Sum')



decoder_df = pd.DataFrame.from_dict(all_decoder_MSEs)

sns.pointplot(x='dataset_id',y='mse', hue='target_type',data=decoder_df)
plt.title('Testing Linear Decoder Performance')
plt.xlabel('Dataset')
plt.ylabel('MSE')
plt.tight_layout()
plt.savefig('results/linear_decoders.png')
plt.close()
