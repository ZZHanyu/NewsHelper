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


# utils
from utils import network
from utils import model


class charactors_hander(model.module):
    def __init__(self, 
                 chunks: pandas.io.parsers.readers.TextFileReader, 
                 main_args,
                 total_len,
                 device) -> None:
        
        print("\nNow inital charactors_hander...\n")
        super().__init__(main_args=main_args,
                         device=device,)
        self.datalen = total_len
        self._meta_data = chunks
        self._LSTM_NetWork = network.trainer(main_args=main_args,
                                             device=device)
        self.embedding_dim = self.embedding_model.vector_size

        print("\ncharactors_hander inital succefuly\n")

        # try:
        #     logging.info(f"Loading models from gensim, name: {main_args.pretrained_embedding_model_name} ....\n")
        #     self.embedding_model = gensim.downloader.load(main_args.pretrained_embedding_model_name)
        # except Exception as e:
        #     print(f"\nERROR! Gensim Loading Failed! \nError contents : {e} \n")
        
        # print(f"** Load Successfully!\n")
        # logging.info(f"** Load Successfully!\n")
        
        # self.tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-uncased')
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self._json_result = {}
        # self.args = main_args


    # def dataset_divider(self):
    #     # length = math.floor(len(self._meta_data))

    #     print(f"\n length = {length}")
        
    #     if self.datalen > 0:
    #         train_num = math.floor(self.datalen * self.args.train_persentage)
    #         self.train_set = self._meta_data[:train_num]
    #         test_num = self.datalen - train_num
    #         self.test_set = self._meta_data[train_num+1:test_num]
    #     else:
    #         logging.info("\n ***** ERROR! Located on dataset_divider!\n")
    #         raise Exception("\n ERROR: Dataset diveded failed! \n")

    def get_classifier_model(self):
        return self._LSTM_NetWork

    def _split_single_word(self) -> list:
        return self._raw_str.split()
    
    def display_elements(self):
        return self._raw_str
    
    def remove_empty_line(self, single_chunk):
        self._remove_empty_line(self, single_chunk=single_chunk)
    
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
                # torch.zeros(self.embedding_dim, dtype=torch.float32 )
        # return torch.tensor(sentences)
        return sentences
    

    def _train(self):
        pass


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
        # Train
        # Epoches
        test_result = []
        for epoch_idx in tqdm(range(self.args.num_epoches), desc="Epoch No.{}".format(epoch_idx), leave=True):
            flag = True # flag == True : Training Progress, flag == False : Testing Progress
            test_tokenized = []

            try:
                # single banch in epoch
                for index, chunk in enumerate(tqdm(self._meta_data, desc="TextFileReader in Progress...", leave=True)):
                    # STEP 1: Remove empty line
                    chunk = self._remove_empty_line(single_chunk=chunk)
                    chunk_tokenized = None # initiaize and re-initiaze

                    # Check whether train progress or test progress
                    if index > math.floor(self.args.train_persentage * index):
                        flag = False

                    # Train chunk
                    if flag == True:
                        for row in tqdm(chunk.iterrows(), leave=True, desc=f"* Train Processing in chunk index {index} ..."):
                            chunk_tokenized.append((self._string_handler(raw_text=row['title'] + row['text']), row['label']))
                        # self._LSTM_NetWork.start(chunk_tokenized, index, flag, num_total_chunk)
                        self._LSTM_NetWork.train(batch=chunk_tokenized, index=index)
                    
                    
                    # Test epoch
                    # -   test need calculate the average accurary
                    # -   average accurary need saved when each epoch finished
                                    
                    elif flag == False:
                        self._LSTM_NetWork.save_model()
                        for row in tqdm(chunk.iterrows(), leave=True, desc=f"* Testing Processing in chunk index {index} ..."):
                            test_tokenized.append((self._string_handler(raw_text=row['title'] + row['text']), row['label']))
                # self._LSTM_NetWork.start(test_tokenized, index, flag, num_total_chunk)
                test_result.append(self._LSTM_NetWork.test(batch=test_tokenized))
            except Exception as e:
                self._LSTM_NetWork.force_save_model()
        
































        
        # # test
        # for index, chunk in enumerate(tqdm(self.train_set, desc="Train in Progress", leave=True)):
        #     # STEP 1: Remove empty line
        #     chunk = self._remove_empty_line(single_chunk=chunk)
        #     chunk_tokenized = [] # initiaize and re-initiaze

        #     for single_index, row in tqdm(chunk.iterrows(), leave=True, desc="Processing in a single chunk..."):
        #         # Concating the news title and main body            
        #         text_merge = row['title'] + row['text']
                
        #         # STEP 2: Lower all charactors in string
        #         text_merge = text_merge.lower()

        #         # STEP 3: Split string into charactor list
        #         # remove all punctuation and split
        #         text_merge = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]','',text_merge).strip()
        #         text_merge = re.findall(r'\b\w+\b', text_merge)

        #         text_merge = self._words_embeddings(text_merge)
        #         if single_index % 13 == 0:
        #             logging.info(f"--> No.{single_index} --> embedding vectors = \n\t{text_merge}\n")
        #         chunk_tokenized.append((text_merge, row['label']))
        # self._LSTM_NetWork.start(chunk_tokenized, index, flag=False)










        # for index, chunk in enumerate(tqdm(self._meta_data, desc="Layer1: Data Preprocess", leave=True)):
        #     # Check wether specifc file number exist:
        #     # if self._file_exist_checker(index):
        #     #     print(f"File tokenized{index} already existed!\n")
        #     #     continue

        #     # STEP 1: Remove empty line
        #     chunk = self._remove_empty_line(single_chunk=chunk)
        #     chunk_tokenized = [] # initiaize and re-initiaze
            
            
        #     for single_index, row in tqdm(chunk.iterrows(), leave=True, desc="Processing in a single chunk..."):
        #         # Concating the news title and main body            
        #         text_merge = row['title'] + row['text']
                
        #         # STEP 2: Lower all charactors in string
        #         text_merge = text_merge.lower()

        #         # remove all punctuation and split
        #         text_merge = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]','',text_merge).strip()
        #         text_merge = re.findall(r'\b\w+\b', text_merge)
                

        #         # STEP 3: Split string into charactor list
        #         #text_merge = text_merge.split()
        #         # print(f"\nHOW many words inside a sequence = {len(text_merge)}\n")
                
        #         # encoder_text = self.tokenizer(text=text_merge, max_length=8, padding='max_length', truncation=True, return_tensors='pt')
                
        #         # print(f"\n The encodered text is = {encoder_text['input_ids']}\n length = {encoder_text['input_ids'].size()} \n")
        #         # temp = np.pad(encoder_text['input_ids'], ((0,0), (0, 512 - encoder_text['input_ids'].shape[1])), mode='constant')
        #         # chunk_tokenized.append((np.pad(encoder_text['input_ids'], ((0,0), (0, 512 - encoder_text['input_ids'].shape[1])), mode='constant'), [row['label']]))
                
        #         #chunk_tokenized.append( (encoder_text['input_ids'], row['label']))
                
        #         #text_merge = torch.tensor(self._words_embeddings(text_merge), requires_grad=True)
        #         text_merge = self._words_embeddings(text_merge)
        #         if single_index % 13 == 0:
        #             logging.info(f"--> No.{single_index} --> embedding vectors = \n\t{text_merge}\n")

        #         chunk_tokenized.append((text_merge, row['label']))
                
        #         # print(f"size after = {temp.shape}")
            
            
        #     self._LSTM_NetWork.start(chunk_tokenized, index)
                


