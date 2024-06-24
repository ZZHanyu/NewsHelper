from transformers import PreTrainedTokenizerFast
import pandas
from tqdm import tqdm
import time 
# import re # wanan to remove other special charactors, ex: [] % ^ @ / ...
# import json
import numpy as np
from datetime import datetime
import os
import logging
import math

import pickle
from numpy import dot
from numpy.linalg import norm

import re
from gensim.models import Word2Vec
import gensim.downloader
import opendatasets as od
import pandas as pd
from abc import ABC, abstractmethod

# for the one hot encoding
import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from tensorflow.keras.preprocessing.text import one_hot
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize


# utils
# from other package
from main_class import main


class data_handler(main): 
    '''
        abstract class - define a customer data loader + handler
    '''

    _datalen = None
    _chunks = None
    _csv_total_len = None
    _chunk_number = None
    _embedding_model = None
    _embedding_dim = None


    @classmethod
    def initialize(cls, batch_size = None):
        cls._dataloader(batch_size)
        cls._datarow_count()


    @classmethod
    def reset(cls, batch_size = None):
        if batch_size == None:
            batch_size = cls._args.chunk_size
        cls._dataloader(batch_size = batch_size)
        

    @classmethod
    def get_generator(cls, batch_size = None):
        if cls._chunks == None:
            cls.initialize()
        data_generator = data_handler_iterator()
        data_iter = iter(data_generator)
        return data_iter
    

    @classmethod
    def _init_embedding_model(cls):
        if cls._embedding_model == None:
            logging.info(f"* Loading pretrained embedding model: {gensim.downloader.info(name=super()._args.pretrained_embedding_model_name)}\n")
            cls._embedding_model = gensim.downloader.load(super()._args.pretrained_embedding_model_name)
            cls._embedding_dim = cls._embedding_model.vector_size
        else:
            pass


    @classmethod
    def _datarow_count(cls):
        with open(super()._args.dataset_path + super()._args.dataset_name + '/WELFake_Dataset.csv') as fileobject:
            cls._csv_total_len = sum(1 for row in fileobject)
        logging.info(f"\n ** DataFile have {cls._csv_total_len} rows of csv data! \n")
        
        # calculate the total chunk number
        cls._chunk_number = cls._csv_total_len // super()._args.chunk_size # this is estimate, because later iterator will delete some row in every chunk
        logging.info(f"\n ** DataFile have {cls._chunk_number} chunks! \n")


    @classmethod
    def _dataloader(cls, batch_size):
        '''
            ---------- Dataset Loader ----------
            Objective:
                - using batchs(chunks) to load csv line by line
            Params:
                - args: chunk size defined
            Return:
                - wda
        '''
        if batch_size == None:
            batch_size = main._args.chunk_size
        else:
            batch_size = batch_size

        if not os.path.exists(super()._args.dataset_path + super()._args.dataset_name + '/WELFake_Dataset.csv'):
            print(f"The file path {super()._args.dataset_path + 'WELFake_Dataset.csv'} is missing! Now downloading...\n")
            # kaggle datasets download -d saurabhshahane/fake-news-classification
            od.download('https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification', data_dir=super()._args.dataset_path)
            cls._chunks = pd.read_csv(super()._args.dataset_path + super()._args.dataset_name + '/' +'WELFake_Dataset.csv', chunksize=batch_size)
        else:
            cls._chunks = pd.read_csv(super()._args.dataset_path + super()._args.dataset_name + '/' +'WELFake_Dataset.csv', chunksize=batch_size)


    @classmethod
    def _remove_empty_line(cls, single_chunk):
        for index, row in single_chunk.iterrows():
            if pandas.isnull(row['title']) or pandas.isnull(row['text']) or pandas.isnull(row['label']):
                single_chunk.drop(index, inplace=True)
            elif type(row['title']) != str or type(row['text']) != str or type(row['label']) != int:
                single_chunk.drop(index, inplace=True)
            elif len(row['title']) < 5 or len(row['text']) < 5:
                single_chunk.drop(index, inplace=True)
        return single_chunk


    @classmethod
    def _normalize(cls, text):
        norm2 = np.linalg.norm(text)
        for j in len(text):
            text[j] = text[j] / norm2
        return text
    

    @classmethod
    def _file_exist_checker(cls, chunk_idx) -> bool:
        if not os.path.exists(super()._args.result_path + "tokenized{}.npy".format(chunk_idx)):
            return False
        else:
            return True
        

    @classmethod   
    def _words_embeddings(cls, sentences):
        if cls._embedding_model == None:
            print("\nEmbedding model missing, NOW Loading...\n")
            cls._init_embedding_model()
        
        for idx in range(len(sentences)):
            if cls._embedding_model.has_index_for(sentences[idx]):
                sentences[idx] = cls._embedding_model.get_vector(sentences[idx], norm=True).astype(np.float32)
            else:
                sentences[idx] = np.zeros(shape=(cls._embedding_dim), dtype=np.float32)

        return sentences
    

    # @classmethod
    # def _one_hot_embeddings(cls, sentences):
    #     lm = WordNetLemmatizer()
    #     # Vocab_size = Unique words in our Corpus (entire document)
    #     vocab_size = 10000

    #     nltk.download('stopwords')
    #     #stopwords = stopwords.words('english')
    #     corpus = []
    #     for i in range(len(sentences)):
    #         review = re.sub('^a-zA-Z0-9',' ',sentences)
    #         review = review.lower()
    #         review = review.split()
    #         review =[lm.lemmatize(x) for x in review if x not in stopwords]
    #         review = " ".join(review)
    #         corpus.append(review)

    #     max_length = max(len(sentence.split()) for sentence in corpus)
    #     print("Maximum sentence length:", max_length)


    #     onehot_repr=[one_hot(words,vocab_size) for words in corpus]
    #     print(onehot_repr[:5])





    @classmethod
    def _string_handler(cls, raw_text):
        # STEP 1: Lower all charactors in string
        raw_text = raw_text.lower()
        # STEP 2: Split string into charactor list
            # remove all punctuation and split
        raw_text = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]','',raw_text).strip()
        raw_text = re.findall(r'\b\w+\b', raw_text)

        # STEP 3: embedding string into vector represeation
        processed_text = cls._words_embeddings(raw_text)
        #processed_text = cls._one_hot_embeddings(raw_text)
        return processed_text
    


    def run(self):
        # only handle data and save to self.tokenized_chunk
        while True:
            chunk_tokenized = [] # initiaize and re-initiaze
            chunk = next(self._chunks, None)
            if chunk == None:
                raise StopIteration
                break
            print(f"\n!Single chunk = {chunk}!\n")
            # STEP 1: Remove empty line
            chunk = self._remove_empty_line(single_chunk=chunk)
            print(f"\nAfter remove empty line = {chunk}\n")  
            for row in tqdm(chunk.iterrows(), desc="Handling single row in a chunk", leave=False):
                chunk_tokenized.append((self._string_handler(raw_text=row['title'] + row['text']), row['label']))
                # [ [01231232] , [21131313],  [2134123232] , .....]
            yield chunk_tokenized
            continue

    







class data_handler_iterator(data_handler):
    '''
    object class
        - An iterator/ implement of abstract class : data handler
    '''

    def __init__(self) -> None:
        print("\nStart iterator building...\n")
        super().__init__()
        self.chunk_idx = 0
        print("\nIterator building Sucessfully!.\n")


    def __iter__(self):
        return self


    def __next__(self):
        # only handle data and save to self.tokenized_chunk
        # atuple = enumerate(next(super()._chunks, None))
        # print(atuple)
        # index = atuple[0]
        # chunk = atuple[1]

        # if not isinstance(chunk, pd.DataFrame):
        #     raise StopIteration
        
        for chunk in super()._chunks:        
            chunk_tokenized = [] # initiaize and re-initiaze
            # STEP 1: Remove empty line
            chunk = super()._remove_empty_line(single_chunk=chunk)
            self.chunk_idx += 1
            for row in tqdm(chunk.itertuples(), desc="Handling single row in a chunk", leave=False):
                # one hot embedding model:
                # chunk_tokenized.append((super()._one_hot_embeddings(sentences=row[2] + row[3]), row[4]))

                # pre-trained embedding model:
                chunk_tokenized.append((super()._string_handler(raw_text=row[2] + row[3]), row[4]))
            return self.chunk_idx, chunk_tokenized
            

    def reset(self):
        super().reset()
            









        # self._tokenized_chunks.append(chunk_tokenized)







        # # Train
        # # Epoches
        # test_result = []
        # for epoch_idx in tqdm(range(self.args.num_epoches), desc="Epoch No.{}".format(epoch_idx), leave=True):
        #     flag = True # flag == True : Training Progress, flag == False : Testing Progress
        #     test_tokenized = []

        #     for chunk_idx, chunk in enumerate(tqdm(self._chunks, desc="TextFileReader in Progress...", leave=True)):
        #         # STEP 1: Remove empty line
        #         chunk = self._remove_empty_line(single_chunk=chunk)
        #         chunk_tokenized = None # initiaize and re-initiaze

        #         # Train chunk
        #         if flag == True:
        #             for single_idx, row in tqdm(chunk.iterrows(), leave=True, desc=f"* Train Processing in chunk index {chunk_idx} ..."):
        #                     # Check whether train progress or test progress
        #                 if single_idx > math.floor(self.args.train_persentage * total_length_num):
        #                     flag = False
        #                 chunk_tokenized.append((self._string_handler(raw_text=row['title'] + row['text']), row['label']))
        #             self._LSTM_NetWork.train(batch=chunk_tokenized, index=chunk_idx)
                
        #         # Test epoch
        #         # -   test need calculate the average accurary
        #         # -   average accurary need saved when each epoch finished
                                
        #         elif flag == False:
        #             self._LSTM_NetWork.save_model()
        #             for row in tqdm(chunk.iterrows(), leave=True, desc=f"* Testing Processing in chunk index {chunk_idx} ..."):
        #                 test_tokenized.append((self._string_handler(raw_text=row['title'] + row['text']), row['label']))
        #     # self._LSTM_NetWork.start(test_tokenized, index, flag, num_total_chunk)
        #     test_result.append(self._LSTM_NetWork.test(batch=test_tokenized))

        
