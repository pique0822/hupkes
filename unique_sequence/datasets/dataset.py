import numpy as np

class Dataset(object):
    """Dataset for the arithmetic language"""

    def __init__(self, dataset_file, vocabulary_file, training_percent=0.6, batch_size=None, seed=None):
        super(Dataset, self).__init__()
        if batch_size is None:
            batch_size = 1

        
        np.random.seed(seed)

        self.dataset_file = dataset_file
        self.training_percent = training_percent

        self.batch_size = batch_size

        # getting the vocabulary
        self.vocabulary = []
        with open(vocabulary_file) as file:
            for line in file:
                self.vocabulary.append(line.strip())
        self.tokenize()

    def tokenize(self):
        # tokenize the dataset
        self.X = []
        self.y = []
        self.pos = []
        with open(self.dataset_file) as file:
            for line in file:
                tokenized_line = []
                positions_in_line = []

                input,solution = line.split(';')
                words = input.split(' ')

                for widx, word in enumerate(words):
                    token = self.word2token(word)
                    tokenized_line.append(token)
                    positions_in_line.append(widx)

                unique_value = self.word2token(solution.strip())

                self.X.append(tokenized_line)
                self.y.append(unique_value)
                self.pos.append(positions_in_line)

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.pos = np.array(self.pos)

        # generate random training set
        self._training_indices = np.random.choice(range(0,self.X.shape[0]),self.X.shape[0],replace=False)

        self._testing_indices = self._training_indices[int(self.training_percent * self.X.shape[0]):]

        self._training_indices = self._training_indices[:int(self.training_percent * self.X.shape[0])]

    def get_vocabulary(self):
        return list(self.vocabulary)

    def set_vocabulary_file(self, vocabulary_file):
        self.vocabulary = []
        with open(vocabulary_file) as file:
            for line in file:
                self.vocabulary.append(line.strip())
        self.tokenize()

    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary
        self.tokenize()

    def word2token(self, word):
        return self.vocabulary.index(word)

    def token2word(self, token):
        return self.vocabulary[token]

    def get_datapoint(self, index, training=True, positions=False):

        if training:
            lower_bound_index = max(0,index*self.batch_size)
            upper_bound_index = min(len(self._training_indices),(index+1)*self.batch_size)

            relevant_indices = self._training_indices[lower_bound_index : upper_bound_index]
        else:
            lower_bound_index = max(0,index*self.batch_size)
            upper_bound_index = min(len(self._testing_indices),(index+1)*self.batch_size)

            relevant_indices = self._testing_indices[lower_bound_index : upper_bound_index]

        X = self.X[relevant_indices,:]
        y = self.y[relevant_indices]
        if positions:
            pos = self.pos[relevant_indices,:]
            return X,y,pos
        return X,y

    def batched_training_size(self):
        if len(self._training_indices) / self.batch_size - int(len(self._training_indices) / self.batch_size) > 0:
            return int(len(self._training_indices) / self.batch_size) + 1
        else:
            return int(len(self._training_indices) / self.batch_size)

    def batched_testing_size(self):
        if len(self._testing_indices) / self.batch_size - int(len(self._testing_indices) / self.batch_size) > 0:
            return int(len(self._testing_indices) / self.batch_size) + 1
        else:
            return int(len(self._testing_indices) / self.batch_size)


if __name__ == '__main__':
    l2 = Dataset('L2/ten_tokens_singular_data.txt','ten_tokens.txt')
    l3 = Dataset('L3/ten_tokens_singular_data.txt','ten_tokens.txt')
