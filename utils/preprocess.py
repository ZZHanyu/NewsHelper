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


# utils

# from other package
from main_class import main


class data_handler(main):
    def __init__(self) -> None:
        print("\nNow inital charactors_hander...\n")
        super().__init__()
        self.datalen = None
        self._chunks = None
        self.csv_total_len = 0
        self.chunk_number = 0
        self._tokenized_chunks = []
        self.embedding_model =  gensim.downloader.load(self.args.pretrained_embedding_model_name)
        self.embedding_dim = self.embedding_model.vector_size
        self._dataloader()
        self.datarow_count()
        self.data_generator = data_handler_iterator()
        print("Charactors_hander inital succefuly")
        

    def get_len_of_total_chunk(self):
        return len(self._tokenized_chunks)


    def datarow_count(self):
        with open(self.args.dataset_path + self.args.dataset_name + '/WELFake_Dataset.csv') as fileobject:
            self.csv_total_len = sum(1 for row in fileobject)
        logging.info(f"\n ** DataFile have {self.csv_total_len} rows of csv data! \n")
        
        # calculate the total chunk number
        chunk_number = self.csv_total_len // self.args.chunk_size # this is estimate, because later iterator will delete some row in every chunk
        logging.info(f"\n ** DataFile have {chunk_number} chunks! \n")


    def _dataloader(self):
        '''
            ---------- Dataset Loader ----------
            Objective:
                - using batchs(chunks) to load csv line by line
            Params:
                - args: chunk size defined
            Return:
                - wda
        '''
        if not os.path.exists(self.args.dataset_path + self.args.dataset_name + '/WELFake_Dataset.csv'):
            print(f"The file path {self.args.dataset_path + 'WELFake_Dataset.csv'} is missing! Now downloading...\n")
            # kaggle datasets download -d saurabhshahane/fake-news-classification
            od.download('https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification', data_dir=self.args.dataset_path)
        else:
            self._chunks = pd.read_csv(self.args.dataset_path + self.args.dataset_name + '/' +'WELFake_Dataset.csv', chunksize=self.args.chunk_size)
    
    
    def display_elements(self):
        return self._raw_str
    
    
    def _remove_empty_line(self, single_chunk):
        for index, row in single_chunk.iterrows():
            if pandas.isnull(row['title']) or pandas.isnull(row['text']) or pandas.isnull(row['label']):
                single_chunk.drop(index, inplace=True)
            elif type(row['title']) != str or type(row['text']) != str or type(row['label']) != int:
                single_chunk.drop(index, inplace=True)
            elif len(row['title']) < 5 or len(row['text']) < 5:
                single_chunk.drop(index, inplace=True)
        return single_chunk

    def _normalize(self, text):
        norm2 = np.linalg.norm(text)
        for j in len(text):
            text[j] = text[j] / norm2
        return text
    

    # def _save_to_file(self, count_num):
    #     with open(self.args.result_path + "tokenized{}.npy".format(str(count_num)), 'a+') as json_file:
    #         json.dump(self._json_result, json_file, sort_keys=True, indent=4)

    def _file_exist_checker(self, chunk_idx) -> bool:
        if not os.path.exists(self.args.result_path + "tokenized{}.npy".format(chunk_idx)):
            return False
        else:
            return True
        
    def _words_embeddings(self, sentences):
        for idx in range(len(sentences)):
            if self.embedding_model.has_index_for(sentences[idx]):
                sentences[idx] = self.embedding_model.get_vector(sentences[idx], norm=True).astype(np.float32)
            else:
                sentences[idx] = np.zeros(shape=(self.embedding_dim), dtype=np.float32)

        return sentences

    def _string_handler(self, raw_text):
        # STEP 1: Lower all charactors in string
        raw_text = raw_text.lower()
        # STEP 2: Split string into charactor list
        # remove all punctuation and split
        raw_text = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]','',raw_text).strip()
        raw_text = re.findall(r'\b\w+\b', raw_text)

        # STEP 3: embedding string into vector represeation
        processed_text = self._words_embeddings(raw_text)
        return processed_text
    

    def arg_max(self):
        pass


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
                print(row)
                chunk_tokenized.append((self._string_handler(raw_text=row['title'] + row['text']), row['label']))
                # [ [01231232] , [21131313],  [2134123232] , .....]
            print(f"\n ALL Done! chunk_tokenized = {chunk_tokenized}\n")
            yield chunk_tokenized
            continue







class data_handler_iterator(data_handler):
    def __init__(self) -> None:
        super().__init__()

    def __iter__(self):
        return self
    
    def __next__(self):
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
                print(row)
                chunk_tokenized.append((self._string_handler(raw_text=row['title'] + row['text']), row['label']))
                # [ [01231232] , [21131313],  [2134123232] , .....]
            print(f"\n ALL Done! chunk_tokenized = {chunk_tokenized}\n")
            yield chunk_tokenized
            continue









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

        
