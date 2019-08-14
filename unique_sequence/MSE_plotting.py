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

import os
if not os.path.exists('results'):
    os.makedirs('results')

parser = argparse.ArgumentParser(description='Trains a GRU model on the arithmetic language dataset')
# model information
parser.add_argument('--embedding_size',type=int, default=2,
                    help='The dimension of the model embedding')
parser.add_argument('--hidden_size',type=int, default=15,
                    help='The dimension of the model hidden state')

# dataset_information
parser.add_argument('--vocabulary_file',type=str,
                    default='datasets/ten_tokens.txt',
                    help='File containing each possible word in the training set.')
parser.add_argument('--dataset_file',type=str, default='ten_tokens_singular_data.txt',
                    help='File that contains the full dataset from which we will take the training and test data.')
parser.add_argument('--test_percent',type=float,
                    default=0.2,
                    help='Percent of data that is to be used as testing.')
# parser.add_argument('--model_save',type=str, default=None,
#                     help='File name to which we will save the model.')
args = parser.parse_args()

def plot_model_MSE(dataset, show=False):
    mses_df = pd.DataFrame.from_dict(dataset)

    sns.pointplot(x='dataset_id',y='acc',data=mses_df)
    plt.title('Testing Model Top 1 Accuracy')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('results/model_perf_all.png')
        plt.close()


    sns.factorplot(x='dataset_id',y='acc',hue='model_id',data=mses_df)
    plt.title('Testing Model Top 1 Accuracy by Model')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
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

relevant_models = 1
relevant_datasets = 10
vocabulary = []
with open(args.vocabulary_file, 'r') as vocab_file:
    for line in vocab_file:
        vocabulary.append(line.strip())

# dataframe
# |  MODELID  |  DATASET  |  MSE  |

# Acc over all datasets models
MSE_per_model = {'model_id':[],'dataset_id':[],'acc':[]}

# acc over all datasets linear decoders
all_decoder_MSEs = {'model_id':[], 'dataset_id':[], 'target_type':[], 'acc':[]}

training_percent = 0.9

for k in range(relevant_models):
    print('Model '+str(k+1))

    model = GatedGRU(input_size = len(vocabulary),
                     embedding_size = args.embedding_size,
                     hidden_size = args.hidden_size,
                     output_size = len(vocabulary))
    model.load_state_dict(torch.load('models/ten_tokens_'+str(k+1)+'.mdl_epoch_50'))
    model.eval()


    for dataset_number in range(2,relevant_datasets+1):
        print('L'+str(dataset_number))

        # loading dataset
        dataset = Dataset('datasets/L'+str(dataset_number)+'/'+args.dataset_file, args.vocabulary_file, 1 - args.test_percent, test_batch_size)

        # getting data
        dataset_correct = 0
        dataset_incorrect = 0
        for data_idx in range(dataset.batched_testing_size()):
            X,y = dataset.get_datapoint(data_idx, training=False)

            hidden = model.init_hidden()

            X = torch.Tensor(X).reshape(test_batch_size, -1, 1).long()
            y = torch.Tensor(y).reshape(-1).long()

            output, hidden, _ = model(X,hidden)

            class_prediction = np.argmax(output.detach().numpy())

            if class_prediction.item() == y.item():
                dataset_correct += 1
            else:
                dataset_incorrect += 1


        MSE_per_model['model_id'].append(k+1)
        MSE_per_model['dataset_id'].append('L'+str(dataset_number))
        MSE_per_model['acc'].append(dataset_correct/(dataset_correct + dataset_incorrect))

        # linear decoders
        #
        # cum_sum = np.array(cumulative_sum_dataset)
        # rec_sum = np.array(recursive_sum_dataset)
        #
        # hiddens = np.array(dataset_hiddens).reshape(len(cum_sum),-1)
        #
        # training_set = np.random.choice(range(len(hiddens)),len(hiddens),replace=True)
        #
        # testing_set = training_set[int(len(training_set)*training_percent):]
        # training_set = training_set[:int(len(training_set)*training_percent)]
        #
        # # cumulative sum
        # reg = LinearRegression()
        # reg.fit(hiddens[training_set,:], cum_sum[training_set])
        #
        # predictions = reg.predict(hiddens[testing_set,:])
        #
        # mse = mean_squared_error(predictions, cum_sum[testing_set])
        #
        # all_decoder_MSEs['model_id'].append(k+1)
        # all_decoder_MSEs['dataset_id'].append('L'+str(dataset_number+1))
        # all_decoder_MSEs['mse'].append(mse)
        # all_decoder_MSEs['target_type'].append('Cumulative Sum')
        #
        # # recursive sum
        # reg = LinearRegression()
        # reg.fit(hiddens[training_set,:], rec_sum[training_set])
        #
        # predictions = reg.predict(hiddens[testing_set,:])
        #
        # mse = mean_squared_error(predictions, rec_sum[testing_set])
        #
        # all_decoder_MSEs['model_id'].append(k+1)
        # all_decoder_MSEs['dataset_id'].append('L'+str(dataset_number+1))
        # all_decoder_MSEs['mse'].append(mse)
        # all_decoder_MSEs['target_type'].append('Recursive Sum')

plot_model_MSE(MSE_per_model, show=True)
# plot_decoder_MSE(all_decoder_MSEs, show=False)
