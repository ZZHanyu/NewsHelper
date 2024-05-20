import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import lda
import lda.datasets
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_texts
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.ldamodel import LdaModel

import opendatasets as od
import pandas as pd
from tqdm import tqdm
import logging
import os
import re
import time

from utils import model
from main_class import main
from utils import preprocess
from utils import network



class LDA_topic_model(model.module, preprocess.data_handler):
    def __init__(self) -> None:
        model.module.__init__(self)
        preprocess.data_handler.initialize()
        self.classify_model = network.LstmNet(  hidden_size = 128,
                                                hidden_size_linear = 64,
                                                num_layers = 2,
                                                dropout = 0,
                                                dropout_linear = 0,
                                                activation_linear = 'LeakyReLU').to(self._device)

        self.data_generator = preprocess.data_handler.get_generator()


    def _init_classification_model(self):
        # load and run classification model
        print("Loading classification model:")
        model_name = self._args.model_save_path + self._load_model()
        if os.path.exists(model_name):
            check_point = torch.load(model_name)
            self.classify_model.load_state_dict(check_point)
        else:
            raise Exception


    def _train_LDA(self):
        common_dictionary = Dictionary(common_texts)
        common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

        lda = LdaModel(
            common_corpus, 
            num_topics=10, 
            alpha='auto', 
            eval_every=5,
            passes=20,
            iterations=100,
            chunksize=2000
            )
        
        while True:
            real_news_list = self._classification()
            if real_news_list == None:
                break
            realnews_corpus = [common_dictionary.doc2bow(text.split()) for text in real_news_list]
            lda.update(realnews_corpus)
        return lda
                    


    def forward(self):
        lda_path = '/Users/taotao/Documents/GitHub/FYP/LDA_Model/'
        if os.path.exists(lda_path):
            if len(os.listdir(lda_path)) == 0:
                # train a basic LDA model
                lda = self._train_LDA()
                self._save_lda(lda)
            else:
                lda = LdaModel.load(lda_path)
        else:
            raise FileNotFoundError


        self._init_classification_model()
        
        topic_model_vector = []
        while True:
            real_news_list = self._classification()
            if real_news_list == None:
                break
            realnews_corpus = [(text, common_dictionary.doc2bow(text.split())) for text in real_news_list]
            for unseen_text in realnews_corpus:
                topic_model_vector.append((unseen_text[0], lda[unseen_text[1]]))
            # lda.update(realnews_corpus)
            print("\nRESULT:\n")
            for i in topic_model_vector:
                print(f"\n{i}\n")
            time.sleep(2)

        return topic_model_vector

    
    def _save_lda(self, lda):
        path = f"{self._args.LDA_model_path}lda_model{self._args.date}"
        lda.save(path)


    def _load_model(self):
        length = len(os.listdir(self._args.model_save_path))
        if os.path.exists(self._args.model_save_path) and length > 1:
            print(f"\n Please choose the a spefic model below:\n")
            for idx, filename in zip(range(len(os.listdir(self._args.model_save_path))), os.listdir(self._args.model_save_path)):
                print(f"{idx}, {filename} \n")
            choose = int(input("\nYour choice :"))
            if choose < length and choose >= 0:
                return filename
        elif length == 1:
            return os.listdir(self._args.model_save_path)[0]


    def _classification(self):
        real_news_list = []
        self.classify_model.eval()
        
        single_chunk = next(preprocess.data_handler._chunks, None)
        if isinstance(single_chunk, pd.DataFrame):
            single_chunk = preprocess.data_handler._remove_empty_line(single_chunk)
            for row in tqdm(single_chunk.itertuples(), desc="Handling single row in a chunk", leave=False):
                temp = preprocess.data_handler._string_handler(raw_text=row[2] + row[3]), row[4]
                feature = torch.tensor(temp[0], dtype=torch.float32, device=main._device, requires_grad=True)
                target = torch.tensor([temp[1]], dtype=torch.float32, device=main._device, requires_grad=True)
                y_pred = self.classify_model.forward(feature)
                if y_pred >= 0.5 and target[0] == 1:
                    real_news_list.append(row[2]+row[3])
                else:
                    continue
            return real_news_list
        else:
            return None
            
        
                    
        




# X.shape
# model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
# model.fit(X)  # model.fit_transform(X) is also available
# topic_word = model.topic_word_  # model.components_ also works
# n_top_words = 8
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))