import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import lda
import lda.datasets
from gensim.test.utils import common_corpus, common_dictionary

import opendatasets as od
import pandas as pd
from tqdm import tqdm
import logging
import os


class LDA_topic_model():
    def __init__(self,
                main_args) -> None:
        # DataLoader
        self.args = main_args
        self._X = lda.datasets.load_reuters()
        self._vocab = lda.datasets.load_reuters_vocab()
        self._titles = lda.datasets.load_reuters_titles()

        # Model identifier
        self.LDA_model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
        self._classifier_model = None
    

        # Fit model parameters' size to dataset size
        self.LDA_model.fit(self._X)
        pass
    
    def forward(self):
        model_name = self.args.model_save_path + self._load_model()
        self._classifier_model = torch.load(model_name)

        topic_word = self.LDA_model.topic_word_
        return topic_word
    
    def _load_model(self):
        length = len(os.listdir(self.args.model_save_path))
        if os.path.exists(self.args.model_save_path) and length > 1:
            print(f"\n Please choose the a spefic model below:\n")
            for idx, filename in zip(range(len(os.listdir(self.args.model_save_path))), os.listdir(self.args.model_save_path)):
                print(f"{idx}, {filename} \n")
            choose = input("\nYour choice :")
            if choose < length and choose >= 0:
                return filename
        elif length == 1:
            return os.listdir(self.args.model_save_path)[0]

    def _classification(self):
        if not os.path.exists(self.args.dataset_path + self.args.dataset_name + '/WELFake_Dataset.csv'):
            print(f"The file path {self.args.dataset_path + 'WELFake_Dataset.csv'} is missing! Now downloading...\n")
            # kaggle datasets download -d saurabhshahane/fake-news-classification
            od.download('https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification', data_dir=self.args.dataset_path)
        else:
            data_chunk = pd.read_csv(self.args.dataset_path+self.args.dataset_name + '/WELFake_Dataset.csv', chunksize=self.args.chunk_size)
        
        self._classifier_model.eval()
        for index, chunk in enumerate(tqdm(data_chunk, desc="TextFileReader in Progress...", leave=True)):
            




X.shape
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))