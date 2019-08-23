from Gated_GRU import GatedGRU
from datasets.dataset import Dataset

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

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

#  python3 acc_plotting.py --vocabulary_file datasets/ten_tokens.txt --base_name ten_tokens --model_directory models_implicit_forget --dataset_file ten_tokens_singular_data.txt

#  python3 acc_plotting.py --vocabulary_file datasets/ten_tokens_explicit.txt --base_name ten_tokens_explicit --model_directory models --dataset_file ten_tokens_explicit_singular_data.txt
#  python3 acc_plotting.py --vocabulary_file datasets/ten_tokens.txt --base_name ten_tokens_repeated --model_directory models --dataset_file ten_tokens_repeated_singular_data.txt

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
parser.add_argument('--base_name',type=str,
                    default='ten_tokens',
                    help='Base name of the models that we will observe')
parser.add_argument('--model_directory',type=str,
                    default='models',
                    help='Folder containing all models that we will observe')
parser.add_argument('--dataset_file',type=str, default='ten_tokens_singular_data.txt',
                    help='File that contains the full dataset from which we will take the training and test data.')
parser.add_argument('--test_percent',type=float,
                    default=0.4,
                    help='Percent of data that is to be used as testing.')
# parser.add_argument('--model_save',type=str, default=None,
#                     help='File name to which we will save the model.')
args = parser.parse_args()

def plot_model_MSE(dataset, show=False):
    mses_df = pd.DataFrame.from_dict(dataset)

    fig = sns.pointplot(x='dataset_id',y='acc',data=mses_df)
    ax = fig.axes
    ax.set_ylim(0,1)
    plt.title('Testing Model Top 1 Accuracy')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('results/model_perf_all.png')
        plt.close()


    fig = sns.factorplot(x='dataset_id',y='acc',hue='model_id',data=mses_df)
    ax = fig.axes
    ax[0,0].set_ylim(0,1)
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

    for mem_idx,df in decoder_df.groupby('mem_idx'):
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

relevant_models = 10
relevant_datasets = 10

forget_delta = 0

vocabulary = []
with open(args.vocabulary_file, 'r') as vocab_file:
    for line in vocab_file:
        vocabulary.append(line.strip())

# dataframe
# |  MODELID  |  DATASET  |  MSE  |

# Acc over all datasets models
MSE_per_model = {'model_id':[],'dataset_id':[],'acc':[]}

# acc over all datasets linear decoders
all_decoder_MSEs = {'model_id':[], 'dataset_id':[], 'target_type':[], 'mem_idx':[], 'acc':[]}

training_percent = 0.9

for k in range(0,relevant_models):
    print('Model '+str(k+1))

    model = GatedGRU(input_size = len(vocabulary),
                     embedding_size = args.embedding_size,
                     hidden_size = args.hidden_size,
                     output_size = len(vocabulary))
    model.load_state_dict(torch.load(args.model_directory+'/'+args.base_name+'_'+str(k+1)+'.mdl_epoch_1990'))
    model.eval()

    for dataset_number in range(2,relevant_datasets+1):
        print('L'+str(dataset_number))

        ingestion_numbers = []

        # loading dataset
        dataset = Dataset('datasets/L'+str(dataset_number)+'/'+args.dataset_file, args.vocabulary_file, 1 - args.test_percent, test_batch_size)

        # getting data
        dataset_hiddens = []
        dataset_updates = []
        dataset_resets  = []

        # 0 will be memory steps, 1 will be forget steps
        transitions = []

        dataset_correct = 0
        dataset_incorrect = 0

        stored_ys = {}
        for data_idx in range(dataset.batched_testing_size()):
            X,y = dataset.get_datapoint(data_idx, training=False)

            stored_index = np.where(X[:,:dataset_number][0] == y)[0].item()

            if stored_index in stored_ys:
                stored_ys[stored_index].append(data_idx)
            else:
                stored_ys[stored_index] = []
                stored_ys[stored_index].append(data_idx)


            for line in X:
                transitions = transitions + [0]*(dataset_number+forget_delta) + [1]*(len(line)-(dataset_number+forget_delta))
                numbers_per_line = []
                for token in line[:dataset_number]:
                    numbers_per_line.append(int(vocabulary[token]))
                ingestion_numbers.append(numbers_per_line)

            hidden = model.init_hidden()

            X = torch.Tensor(X).reshape(test_batch_size, -1, 1).long()
            y = torch.Tensor(y).reshape(-1).long()


            output, hidden, (update_gates, reset_gates, hidden_states) = model(X,hidden)

            line_hiddens = []
            line_updates = []
            line_resets = []
            for t in range(len(hidden_states)):
                hidden_t = hidden_states[t].reshape(-1)
                update_t = update_gates[t].reshape(-1)
                reset_t = reset_gates[t].reshape(-1)

                line_hiddens.append(hidden_t.detach().numpy())
                line_updates.append(update_t.detach().numpy())
                line_resets.append(reset_t.detach().numpy())

            dataset_hiddens.append(line_hiddens)
            dataset_updates.append(line_updates)
            dataset_resets.append(line_resets)

            class_prediction = np.argmax(output.detach().numpy())

            if class_prediction.item() == y.item():
                dataset_correct += 1
            else:
                dataset_incorrect += 1


        MSE_per_model['model_id'].append(k+1)
        MSE_per_model['dataset_id'].append('L'+str(dataset_number))
        MSE_per_model['acc'].append(dataset_correct/(dataset_correct + dataset_incorrect))

        # All memories by index
        seq_nums = np.array(ingestion_numbers)
        trans_classes = np.array(transitions)

        hiddens = np.array(dataset_hiddens)
        updates = np.array(dataset_updates)
        resets = np.array(dataset_resets)



        # Statistical difference between memory and forget?
        sequential_hiddens = np.concatenate(hiddens, 0)
        sequential_updates = np.concatenate(updates, 0)
        sequential_resets = np.concatenate(resets, 0)


        training_set = np.random.choice(range(len(sequential_hiddens)),len(sequential_hiddens),replace=True)

        testing_set = training_set[int(len(training_set)*training_percent):]
        training_set = training_set[:int(len(training_set)*training_percent)]


        # Transition probabilities in hidden
        svm = LinearSVC()
        cls = CalibratedClassifierCV(svm)
        cls.fit(sequential_hiddens[training_set,:], trans_classes[training_set])


        acc = cls.score(sequential_hiddens[testing_set,:], trans_classes[testing_set])
        print('Hidden Transition Accuracy', acc)

        probs_forget = []
        for training_time_step in range(hiddens.shape[1]):
            hidden_t = hiddens[:,training_time_step,:]

            mean_forget_step_prob = cls.predict_proba(hidden_t)[:,1].mean(0)
            probs_forget.append(mean_forget_step_prob)

        plt.plot(range(hiddens.shape[1]), probs_forget)
        plt.title('Hidden State\nProbability of Forget Mode Over Time')
        plt.xticks(range(hiddens.shape[1]),range(hiddens.shape[1]))
        plt.ylim(0,1)
        plt.xlabel('Timestep')
        plt.ylabel('Probability')


        plt.axvline(x=dataset_number + forget_delta -1, color='r',linestyle='--')
        # plt.show()
        plt.close()




        # transition probabilities in Updates
        svm = LinearSVC()
        cls = CalibratedClassifierCV(svm)
        cls.fit(sequential_updates[training_set,:], trans_classes[training_set])

        acc = cls.score(sequential_updates[testing_set,:], trans_classes[testing_set])
        print('Update Transition Accuracy', acc)

        probs_forget = []
        for training_time_step in range(hiddens.shape[1]):
            hidden_t = updates[:,training_time_step,:]

            mean_forget_step_prob = cls.predict_proba(hidden_t)[:,1].mean(0)
            probs_forget.append(mean_forget_step_prob)

        plt.plot(range(hiddens.shape[1]), probs_forget)
        plt.title('Update Gate\nProbability of Forget Mode Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Probability')
        plt.xticks(range(hiddens.shape[1]),range(hiddens.shape[1]))
        plt.ylim(0,1)
        plt.axvline(x=dataset_number-1, color='r',linestyle='--')
        # plt.show()
        plt.close()


        # transition probabilities in Updates
        svm = LinearSVC()
        cls = CalibratedClassifierCV(svm)
        cls.fit(sequential_resets[training_set,:], trans_classes[training_set])

        acc = cls.score(sequential_resets[testing_set,:], trans_classes[testing_set])
        print('Reset Transition Accuracy', acc)

        probs_forget = []
        for training_time_step in range(hiddens.shape[1]):
            hidden_t = resets[:,training_time_step,:]

            mean_forget_step_prob = cls.predict_proba(hidden_t)[:,1].mean(0)
            probs_forget.append(mean_forget_step_prob)

        plt.plot(range(hiddens.shape[1]), probs_forget)
        plt.title('Reset Gate\nProbability of Forget Mode Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Probability')
        plt.xticks(range(hiddens.shape[1]),range(hiddens.shape[1]))
        plt.ylim(0,1)
        plt.axvline(x=dataset_number-1, color='r',linestyle='--')
        # plt.show()
        plt.close()

        continue
        # # sequence_nums
        for memory_idx in range(dataset_number):

            generalization_matrix = []
            for training_time_step in range(hiddens.shape[1]):
                accuracy_over_time = []
                reg = LinearSVC()
                reg.fit(hiddens[training_set,training_time_step,:], seq_nums[training_set,memory_idx])

                for testing_time_step in range(hiddens.shape[1]):
                    predictions = reg.predict(hiddens[testing_set,testing_time_step,:])

                    acc = accuracy_score(predictions, seq_nums[testing_set,memory_idx])

                    accuracy_over_time.append(acc)

                    all_decoder_MSEs['model_id'].append(k+1)
                    all_decoder_MSEs['dataset_id'].append('L'+str(dataset_number))
                    all_decoder_MSEs['acc'].append(acc)
                    all_decoder_MSEs['mem_idx'].append(memory_idx)
                    all_decoder_MSEs['target_type'].append('Memory Sequence')

                generalization_matrix.append(accuracy_over_time)

            # plt.title('Generaliztion of L'+str(dataset_number)+' on Memory Index '+str(memory_idx+1))
            # plt.imshow(generalization_matrix,vmin=0,vmax=1,origin='lower')
            # plt.colorbar()
            # plt.show()

        # All memories by stored to end

        for key in sorted(stored_ys.keys()):
            stored_indices = stored_ys[key]

            training_set = np.random.choice(stored_indices,len(stored_indices),replace=True)

            testing_set = training_set[int(len(training_set)*training_percent):]
            training_set = training_set[:int(len(training_set)*training_percent)]

            generalization_matrix = []
            for training_time_step in range(hiddens.shape[1]):
                accuracy_over_time = []
                reg = LinearSVC()
                reg.fit(hiddens[training_set,training_time_step,:], seq_nums[training_set,key])

                for testing_time_step in range(hiddens.shape[1]):
                    predictions = reg.predict(hiddens[testing_set,testing_time_step,:])

                    acc = accuracy_score(predictions, seq_nums[testing_set,key])

                    accuracy_over_time.append(acc)

                generalization_matrix.append(accuracy_over_time)

            plt.title('Generaliztion of L'+str(dataset_number)+' when Stored y at Index '+str(key))
            plt.imshow(generalization_matrix,vmin=0,vmax=1,origin='lower')
            plt.colorbar()
            plt.show()
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
# plot_decoder_MSE(all_decoder_MSEs, show=True)
