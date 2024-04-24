import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import lda
import lda.datasets
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.ldamodel import LdaModel

import opendatasets as od
import pandas as pd
from tqdm import tqdm
import logging
import os
import re

from utils import model



class LDA_topic_model(model.module):
    def __init__(self,
                main_args,
                device,
                classifier_model) -> None:
        
        super().__init__(main_args=main_args,
                         device=device,
                         )

        # # DataLoader
        self._classifier_model = classifier_model

        
        # self._X = lda.datasets.load_reuters()
        # self._vocab = lda.datasets.load_reuters_vocab()
        # self._titles = lda.datasets.load_reuters_titles()

        # # Model identifier
        # self.LDA_model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
    

        # # Fit model parameters' size to dataset size
        # self.LDA_model.fit(self._X)

    def forward(self):
        # load and run classification model
        print("Loading classification model:")
        model_name = self.args.model_save_path + self._load_model()
        if os.path.exists(model_name):
            check_point = torch.load(model_name)
            self._classifier_model = self._classifier_model.load_state_dict(check_point)
        else:
            raise Exception
        real_news_list = self._classification()

        # train an LDA model
        # topic_word = self.LDA_model.topic_word_
        # realnews_corpus = [common_dictionary.doc2bow(text) for text in real_news_list]
        print(f"\nTrain LDA start\n")
        lda = LdaModel(common_corpus, num_topics=10)
        print(f"\nTrain LDA sucessfully\n")
        self._save_lda(lda)
        print(f"\n save done\n")

        realnews_corpus = [common_dictionary.doc2bow(text) for text in real_news_list]
        topic_model_vector = []
        for unseen_text in realnews_corpus:
            topic_model_vector.append(lda(unseen_text))
        
        print("\nRESULT:\n")
        for i in topic_model_vector:
            print(f"\n{i}\n")

        return topic_model_vector
    

    def _remove_empty_line(self, single_chunk):
        for index, row in single_chunk.iterrows():
            if pd.isnull(row['title']) or pd.isnull(row['text']) or pd.isnull(row['label']):
                single_chunk.drop(index, inplace=True)
            elif type(row['title']) != str or type(row['text']) != str or type(row['label']) != int:
                single_chunk.drop(index, inplace=True)
            elif len(row['title']) < 5 or len(row['text']) < 5:
                single_chunk.drop(index, inplace=True)
        return single_chunk
    
    
    def _save_lda(self, lda):
        path = f"{self.args.LDA_model_path}lda_model{self.args.date_time}"
        lda.save(path)


    def _load_model(self):
        length = len(os.listdir(self.args.model_save_path))
        if os.path.exists(self.args.model_save_path) and length > 1:
            print(f"\n Please choose the a spefic model below:\n")
            for idx, filename in zip(range(len(os.listdir(self.args.model_save_path))), os.listdir(self.args.model_save_path)):
                print(f"{idx}, {filename} \n")
            choose = int (input("\nYour choice :"))
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
        
        real_news_list = []
        self._classifier_model.eval()
        for index, chunk in enumerate(tqdm(data_chunk, desc="TextFileReader in Progress...", leave=True)):
            chunk = self._remove_empty_line(single_chunk=chunk)
            for single_index, row in tqdm(chunk.iterrows(), leave=True, desc=f"* Testing Processing in chunk index {index} ..."):
                # Concating the news title and main body            
                text_merge = row['title'] + row['text']
                # STEP 2: Lower all charactors in string
                text_merge = text_merge.lower()
                # STEP 3: Split string into charactor list
                # remove all punctuation and split
                text_merge = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]','',text_merge).strip()
                text_merge = re.findall(r'\b\w+\b', text_merge)
                text_merge = self._words_embeddings(text_merge)
                text_merge_tensor = torch.tensor(text_merge, requires_grad=True, device=self.device, dtype=torch.float32)
                y_pred = self._classifier_model(text_merge_tensor)
                if y_pred == row['label']:
                    real_news_list.append(text_merge)
                else:
                    pass

        return real_news_list
                    
        




# X.shape
# model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
# model.fit(X)  # model.fit_transform(X) is also available
# topic_word = model.topic_word_  # model.components_ also works
# n_top_words = 8
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))