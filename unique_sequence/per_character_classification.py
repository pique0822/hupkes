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
                    default='models_implicit_forget',
                    help='Folder containing all models that we will observe')
parser.add_argument('--dataset_file',type=str, default='ten_tokens_singular_data.txt',
                    help='File that contains the full dataset from which we will take the training and test data.')
parser.add_argument('--model_type',type=str, default='implicit',
help='Designates the type of model we will be running.')
parser.add_argument('--graphs',type=str, default='temporal',
help='{temporal | positional | character}')
parser.add_argument('--graph_idx',type=int, default=-1,
help='If left as -1, the index will be random otherwise specify an index in particular')
args = parser.parse_args()

vocabulary = []
with open(args.vocabulary_file, 'r') as vocab_file:
    for line in vocab_file:
        vocabulary.append(line.strip())

colors = ['#1abc9c','#27ae60','#3498db','#9b59b6','#f1c40f','#d35400','#e74c3c','#7f8c8d','#e84393','#9980FA','#34495e']

def generate_char_probability_graph(hiddens, classifiers):
    if args.graph_idx == -1:
        random_index = np.random.randint(len(hiddens))
    else:
        random_index = args.graph_idx

    random_text = text[random_index]
    random_example = hiddens[random_index]
    random_output = outputs[random_index]

    temporal_probabilities = []
    for training_time_step in range(len(random_example)):
        hidden_t = random_example[training_time_step,:].reshape(1,-1)

        character_probabilities = []
        for char_idx, char in enumerate(vocabulary):
            cls = classifiers[char_idx]
            if cls is not None:
                prob = cls.predict_proba(hidden_t)[:,1]
            else:
                prob = 0

            character_probabilities.append(prob)

        temporal_probabilities.append(character_probabilities)

    temporal_probabilities = np.array(temporal_probabilities)
    temporal_probabilities = temporal_probabilities.reshape(temporal_probabilities.shape[0],temporal_probabilities.shape[1]).T

    for char_idx, char in enumerate(vocabulary):
        probabilities = temporal_probabilities[char_idx]
        line = plt.plot(range(len(probabilities)), probabilities, label=char, c=colors[char_idx])

    plt.title('Final Prediction : '+str(random_output))
    plt.xlabel('Timestep')
    plt.ylabel('Probability')
    plt.xticks(range(len(probabilities)), random_text)
    plt.legend()
    plt.show()


def generate_temp_probability_graph(hiddens, temp_classifiers):
    if args.graph_idx == -1:
        random_index = np.random.randint(len(hiddens))
    else:
        random_index = args.graph_idx
    random_text = text[random_index]
    random_example = hiddens[random_index]
    random_output = outputs[random_index]

    temporal_probabilities = []
    for training_time_step in range(len(random_example)):
        hidden_t = random_example[training_time_step,:].reshape(1,-1)

        classifiers = temp_classifiers[training_time_step]
        character_probabilities = []
        for char_idx, char in enumerate(vocabulary):
            cls = classifiers[char_idx]
            if cls is not None:
                prob = cls.predict_proba(hidden_t)[:,1]
            else:
                prob = 0

            character_probabilities.append(prob)

        temporal_probabilities.append(character_probabilities)

    temporal_probabilities = np.array(temporal_probabilities)
    temporal_probabilities = temporal_probabilities.reshape(temporal_probabilities.shape[0],temporal_probabilities.shape[1]).T

    for char_idx, char in enumerate(vocabulary):
        probabilities = temporal_probabilities[char_idx]
        line = plt.plot(range(len(probabilities)), probabilities, label=char, c=colors[char_idx])

    plt.title('Final Prediction : '+str(random_output))
    plt.xlabel('Timestep')
    plt.ylabel('Probability')
    plt.xticks(range(len(probabilities)), random_text)
    plt.legend()
    plt.show()


def generate_pos_probability_graph(hiddens, classifiers):
    if args.graph_idx == -1:
        random_index = np.random.randint(len(hiddens))
    else:
        random_index = args.graph_idx

    random_text = text[random_index]
    random_example = hiddens[random_index]
    random_output = outputs[random_index]

    temporal_probabilities = []
    for training_time_step in range(len(random_example)):
        hidden_t = random_example[training_time_step,:].reshape(1,-1)

        positional_probabilities = []
        for pos_idx in range(len(classifiers)):
            cls = classifiers[pos_idx]
            prob = cls.predict_proba(hidden_t)[:,1]

            positional_probabilities.append(prob)

        temporal_probabilities.append(positional_probabilities)

    temporal_probabilities = np.array(temporal_probabilities)
    temporal_probabilities = temporal_probabilities.reshape(temporal_probabilities.shape[0],temporal_probabilities.shape[1]).T

    for pos_idx in range(len(classifiers)):
        probabilities = temporal_probabilities[pos_idx]
        line = plt.plot(range(len(probabilities)), probabilities, label=str(pos_idx+1), c=colors[pos_idx])

    plt.title('Final Prediction : '+str(random_output))
    plt.xlabel('Timestep')
    plt.ylabel('Probability')
    plt.xticks(range(len(probabilities)), random_text)
    plt.legend()
    plt.show()




model = GatedGRU(input_size = len(vocabulary),
                 embedding_size = args.embedding_size,
                 hidden_size = args.hidden_size,
                 output_size = len(vocabulary))
model.load_state_dict(torch.load(args.model_directory+'/'+args.base_name+'_3.mdl_epoch_2990'))
model.eval()

test_batch_size = 1
training_percent = .9
for dataset_number in range(5,6):
    print('L'+str(dataset_number))

    ingestion_numbers = []

    # loading dataset
    dataset = Dataset('datasets/L'+str(dataset_number)+'/'+args.dataset_file, args.vocabulary_file, 0, test_batch_size, seed=1111)

    # getting data
    dataset_hiddens = []
    dataset_updates = []
    dataset_resets  = []
    dataset_predictions = []

    # 0 will be memory steps, 1 will be forget steps
    transitions = []

    dataset_correct = 0
    dataset_incorrect = 0

    dataset_character_presence = {}
    dataset_position_presence = {}
    dataset_temporal_presence = {}

    if args.model_type == 'explicit':
        for timestep in range(dataset_number*2):
            dataset_temporal_presence[timestep] = {}
    else:
        for timestep in range(dataset_number*2 - 1):
            dataset_temporal_presence[timestep] = {}

    for character in vocabulary:
        dataset_character_presence[character] = []

    for pos_idx in range(dataset_number):
        dataset_position_presence[pos_idx] = []

    text = []
    for data_idx in range(dataset.batched_testing_size()):
        X,y = dataset.get_datapoint(data_idx, training=False)

        for line in X:
            line_text = []

            character_presence = [0]*len(vocabulary)
            positional_presence = [0]*dataset_number
            positional_character = []
            remember = True
            for token_idx, token in enumerate(line):
                character = dataset.token2word(token)
                character_index = vocabulary.index(character)

                if args.model_type == 'explicit':
                    if token_idx == dataset_number:
                        remember = False
                    elif remember:
                        positional_character.append(character)
                        positional_index = token_idx

                        positional_presence[positional_index] = \
                        (positional_presence[positional_index] + 1) % 2
                    else:
                        positional_index = positional_character.index(character)

                        positional_presence[positional_index] = \
                        (positional_presence[positional_index] + 1) % 2
                else:
                    if token_idx < dataset_number:
                        positional_character.append(character)
                        positional_index = token_idx

                        positional_presence[positional_index] = \
                        (positional_presence[positional_index] + 1) % 2
                    else:
                        positional_index = positional_character.index(character)

                        positional_presence[positional_index] = \
                        (positional_presence[positional_index] + 1) % 2

                character_presence[character_index] = (character_presence[character_index] + 1) % 2


                for vocab_idx, vocab_char in enumerate(vocabulary):
                    dataset_character_presence[vocab_char].append(character_presence[vocab_idx])

                for pos_idx in range(len(positional_presence)):
                    dataset_position_presence[pos_idx].append(positional_presence[pos_idx])

                for vocab_idx, vocab_char in enumerate(vocabulary):
                    if vocab_char not in dataset_temporal_presence[token_idx]:
                        dataset_temporal_presence[token_idx][vocab_char] = []

                    dataset_temporal_presence[token_idx][vocab_char].append(character_presence[vocab_idx])

                line_text.append(character)

            text.append(line_text)

        hidden = model.init_hidden()

        X = torch.Tensor(X).reshape(test_batch_size, -1, 1).long()
        y = torch.Tensor(y).reshape(-1).long()


        output, hidden, (update_gates, reset_gates, hidden_states) = model(X,hidden)

        class_prediction = np.argmax(output.detach().numpy())

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
        dataset_predictions.append(class_prediction)


    hiddens = np.array(dataset_hiddens)
    updates = np.array(dataset_updates)
    resets = np.array(dataset_resets)
    outputs = np.array(dataset_predictions)

    if args.graphs == 'temporal':
        # training unique classifiers for each hiddens
        training_set = np.random.choice(range(len(hiddens)),len(hiddens),replace=False)

        testing_set = training_set[int(len(training_set)*training_percent):]
        training_set = training_set[:int(len(training_set)*training_percent)]

        # Each line of dataset K has 2K chars (explicit) or 2K-1 chars (implicit)
        temporal_classifiers = []

        for timestep in range(hiddens.shape[1]):
            training_x = hiddens[training_set,timestep,:]
            testing_x = hiddens[testing_set,timestep,:]

            timestep_classifiers = []
            for vocab_idx, vocab_char in enumerate(vocabulary):
                training_y = np.array(dataset_temporal_presence[timestep][vocab_char])[training_set]
                testing_y = np.array(dataset_temporal_presence[timestep][vocab_char])[testing_set]

                try:
                    svm = LinearSVC()
                    cls = CalibratedClassifierCV(svm)
                    cls.fit(training_x, training_y)

                    accuracy = cls.score(testing_x, testing_y)
                    print('Timestep '+ str(timestep) +' Character '+vocab_char + ' Test Accuracy: ',accuracy)

                    timestep_classifiers.append(cls)
                except:
                    timestep_classifiers.append(None)

            temporal_classifiers.append(timestep_classifiers)


        generate_temp_probability_graph(hiddens, temporal_classifiers)

    if args.graphs == 'character':

        # Classifiers per character
        sequential_hiddens = np.concatenate(hiddens, 0)

        training_set = np.random.choice(range(len(sequential_hiddens)),len(sequential_hiddens),replace=False)

        testing_set = training_set[int(len(training_set)*training_percent):]
        training_set = training_set[:int(len(training_set)*training_percent)]

        character_classifiers = []

        training_x = sequential_hiddens[training_set,:]
        testing_x = sequential_hiddens[testing_set,:]

        # character classifiers
        for vocab_idx, vocab_char in enumerate(vocabulary):
            training_y = np.array(dataset_character_presence[vocab_char])[training_set]
            testing_y = np.array(dataset_character_presence[vocab_char])[testing_set]
            svm = LinearSVC()
            cls = CalibratedClassifierCV(svm)
            cls.fit(training_x, training_y)

            accuracy = cls.score(testing_x, testing_y)
            print('Character '+vocab_char + ' Test Accuracy: ',accuracy)

            character_classifiers.append(cls)
        generate_char_probability_graph(hiddens, character_classifiers)


    if args.graphs == 'positional':
        # Classifiers per character
        sequential_hiddens = np.concatenate(hiddens, 0)

        training_set = np.random.choice(range(len(sequential_hiddens)),len(sequential_hiddens),replace=False)

        testing_set = training_set[int(len(training_set)*training_percent):]
        training_set = training_set[:int(len(training_set)*training_percent)]

        positional_classifiers = []

        training_x = sequential_hiddens[training_set,:]
        testing_x = sequential_hiddens[testing_set,:]
    # positional classifiers
        for pos_idx in range(dataset_number):
            training_y = np.array(dataset_position_presence[pos_idx])[training_set]
            testing_y = np.array(dataset_position_presence[pos_idx])[testing_set]
            svm = LinearSVC()
            cls = CalibratedClassifierCV(svm)
            cls.fit(training_x, training_y)

            accuracy = cls.score(testing_x, testing_y)
            print('Position '+ str(pos_idx+1) + ' Test Accuracy: ',accuracy)

            positional_classifiers.append(cls)

        generate_pos_probability_graph(hiddens, positional_classifiers)