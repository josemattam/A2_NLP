# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier, nn.Module):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, word_embeddings: WordEmbeddings, num_inputs, num_outputs, hidden_size):
        super(NeuralSentimentClassifier, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

        self.word_embeddings = word_embeddings
        self.init_embedding = word_embeddings.get_initialized_embedding_layer(frozen= True)


    def forward(self, input_indices):
        embed_vecs = self.init_embedding(input_indices)
        averaged_embeddings = torch.mean(embed_vecs, dim= 0)

        linear = self.linear1(averaged_embeddings)
        relu = F.relu(linear)
        prediction = self.linear2(relu)
        return prediction

    """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
    """
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        softmax = nn.LogSoftmax(dim=0)
        word_indices = get_word_indices(ex_words, self.word_embeddings.word_indexer)
        output = self.forward(word_indices)

        return torch.argmax(softmax(output)).item()

# added
def get_word_indices(sentence:List[str], indexer:Indexer):
    word_indices = [indexer.index_of(word) if indexer.index_of(word) != -1 else indexer.index_of("UNK") for word in sentence]
 
    return torch.tensor(word_indices, dtype=torch.long)

"""
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
"""
def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    model = NeuralSentimentClassifier(word_embeddings, word_embeddings.get_embedding_length(), args.hidden_size, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    loss = torch.nn.CrossEntropyLoss()     
    for epoch in range(args.num_epochs):
            ex_indices = [i for i in range(0, len(train_exs))]
            random.shuffle(ex_indices)
            total_loss = 0.0

            for idx in ex_indices:
                sentence = train_exs[idx].words
                label = train_exs[idx].label


                word_indices = get_word_indices(sentence, word_embeddings.word_indexer)
                output = model(word_indices)
                loss_val = loss(output.view(1,-1), torch.tensor([label]))
                total_loss += loss_val
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

            print("total loss for this epoch %i: %f" % (epoch, total_loss))
    return model
