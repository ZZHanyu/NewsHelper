from transformers import PreTrainedTokenizerFast
import pandas
from tqdm import tqdm
import time 
import re # wanan to remove other special charactors, ex: [] % ^ @ / ...
import json
import numpy as np
from datetime import datetime

# utils
# from classifier import main
import os

class charactors_hander():
    def __init__(self, chunks: pandas.io.parsers.readers.TextFileReader, main_args) -> None:
        # super().__init__(main_args)
        self._meta_data = chunks
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-uncased')
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self._json_result = {}
        self.args = main_args
        
    # def _remove_empty(self, chunk, *args):
    #     for arg in args:
    #         chunk = chunk[len(chunk[arg]) != 0]

    def _split_single_word(self) -> list:
        return self._raw_str.split()
    
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
    
    def _dataset_divder(self):
        pass

    def _save_to_file(self, count_num):
        # current_time = datetime.now()
        # formatted_time = current_time.strftime("%Y_%m_%d_%H")
        with open(self.args.result_path + "tokenized{}.json".format(str(count_num)), 'a+') as json_file:
            json.dump(self._json_result, json_file, sort_keys=True, indent=4)

    def _file_exist_checker(self, chunk_idx) -> bool:
        if not os.path.exists(self.args.result_path + "tokenized{}.json".format(chunk_idx)):
            return False
        else:
            return True


    def run(self):
        for index, chunk in enumerate(tqdm(self._meta_data, desc="Layer1: Data Preprocess", leave=True)):
            # Check wether specifc file number exist:
            if self._file_exist_checker(index):
                print(f"File tokenized{index} already existed!\n")
                continue

            # STEP 1: Remove empty line (Remove dirty data)
            chunk = self._remove_empty_line(single_chunk=chunk)
            #target_arr = np.array()
            # chunk_tokenized = []
            for single_index, row in tqdm(chunk.iterrows(), leave=True, desc="Processing in a single chunk..."):
                # Concating the news title and main body            
                text_merge = row['title'] + row['text']
                
                # STEP 2: Lower all charactors in string
                text_merge = text_merge.lower()
                
                # STEP 3: Split string into charactor list
                # text_merge = text_merge.split()

                encoder_text = self.tokenizer(text=text_merge, padding=True, truncation=True, return_tensors='pt')
                print(f"\n ** Size of encoder text = {encoder_text['input_ids']} \n")
                #target_arr 
                # chunk_tokenized.append(text_merge)


            # print(f"{type(chunk_tokenized)}",
            #      f"{len(chunk_tokenized)}" )
            # Tokenizer
            # encoder_text = self.tokenizer(text=chunk_tokenized, padding=True, truncation=True, return_tensors='pt')
            # self._json_result[single_index] = (encoder_text['input_ids'].numpy().tolist(), row['label'])
            
            # print(f"\n **** Length = {len(self._json_result)} \n ")
            # # Memory Saver
            # self._save_to_file(count_num=index) 
            # self._json_result = {} # release memory

            
            # Need to solve : High memory usage!
            # Memory Saver
            # if index % 1000 == 0 and index != 0:
                
            #     self._save_to_file(count_num=index) 
            #     self._json_result = {} # release memory
            #     self._json_result[index] = (encoder_text['input_ids'].numpy().tolist(), row['label'])
            # else:
            #     self._json_result[index] = (encoder_text['input_ids'].numpy().tolist(), row['label'])
            # # time.sleep(5)

        print(self._json_result)

        # try:
        #     self._save_to_file(count_num=index//1000)
        #     print(f"\nSave to json file succeed!\n")
        # except Exception as e:
        #     print(f"\nERROR! save file failed, {e}")
        #     #print(f"**After modify : \t title = {row['title']}, context = {row['text']}, label = {row['label']} \n\n")


        # except Exception as e:
        #     print(f"Data Preprocessing Failed! error is {e}")


        
        # if self._empty_judge() == False:
        #     assert 1==0, print("** ERROR! there are blank line exist")
        
        


        # split into single words set



        #encoded_inputs = self.tokenizer(self._raw_str, padding=True, return_tensors='pt')
        #print(encoded_inputs)


    

