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

    def _save_to_file(self):
        # current_time = datetime.now()
        # formatted_time = current_time.strftime("%Y_%m_%d_%H")
        with open(self.args.result_path + "tokenized" + self.args.date_time, 'a+') as json_file:
            json.dump(self._json_result, json_file, sort_keys=True, indent=4)

    def run(self):
        json_path = self.args.result_path + "tokenized" + self.args.date_time
        if not os.path.exists(json_path):
            for chunk in tqdm(self._meta_data, desc="Layer1: Data Preprocess", leave=True ):
                # Step1: check whether empty
                    # if a blank dectected, then delete this line
                # self._remove_empty(chunk, 'title', 'text', 'label')

                # STEP 1: Remove empty line (Remove dirty data)
                chunk = self._remove_empty_line(single_chunk=chunk)
                # chunk = chunk[(chunk['title'].notnull()) & (chunk['title'].dtypes == object)]
                # chunk = chunk[(chunk['text'].apply(lambda x: True if (x != None) & (type(x) == str) else False))]
                # chunk = chunk[(chunk['label'].notnull()) & (chunk['label'].dtypes == int)]

                for index, row in tqdm(chunk.iterrows(), leave=True, desc="Processing in a single chunk..."):
                    #print(f"index = {index}\n context = {row}, \ntitle = {row['title']}\n type = {type(row['title'])},\n length={len(row['title'])} \n")
                                
                    text_merge = row['title'] + row['text']
                    # STEP 2: Lower all charactors in string
                    # row['title'] = row['title'].lower()
                    # row['text'] = row['text'].lower()
                    text_merge = text_merge.lower()
                    # print(f"{len(row['text'])}\n")
                    
                    # split
                    # row['title'] = row['title'].split()
                    # row['text'] = row['text'].split()
                    text_merge = text_merge.split()

                    # tokenizer
                    # print(f"{row['text']} {len(row['text'])} {type(row['text'])}\n")
                    # encoder_title = self.tokenizer(row['title'], padding=True, return_tensors='pt')
                    # encoder_text = self.tokenizer(row['text'], padding=True,  return_tensors='pt')
                    encoder_text = self.tokenizer(text_merge, padding=True,  return_tensors='pt')
                    
                    #print(f"token result = text = {encoder_text} \n {encoder_text['input_ids'].size()}\n\n")
                    #time.sleep(5)
                    # print(f"\n Token result = {encoder_text} \n Size = {encoder_text['input_ids'].size()}")

                    # L2 normlization: (NO NEED)
                    # encoder_text = self._normalize(encoder_text)
                    # print(f"\n\n ********** After norm2, text = {encoder_text} \nSize text = {encoder_text.size()} ")
                    # Need to solve : High memory usage!
                    
                    # Memory Saver
                    if index % 1000 == 0:
                        print(f"\n **** Length = {len(self._json_result)} \n ")
                        self._save_to_file()
                        self._json_result = {} # release memory
                        self._json_result[index] = (encoder_text['input_ids'].numpy().tolist(), row['label'])
                    else:
                        self._json_result[index] = (encoder_text['input_ids'].numpy().tolist(), row['label'])
                    # time.sleep(5)

            print(self._json_result)
            try:
                self._save_to_file()
                print(f"\nSave to json file succeed!\n")
            except Exception as e:
                print(f"\nERROR! save file failed, {e}")
                #print(f"**After modify : \t title = {row['title']}, context = {row['text']}, label = {row['label']} \n\n")
        else:
            print(f"\n File Existed! \n")
            pass

        # except Exception as e:
        #     print(f"Data Preprocessing Failed! error is {e}")


        
        # if self._empty_judge() == False:
        #     assert 1==0, print("** ERROR! there are blank line exist")
        
        


        # split into single words set



        #encoded_inputs = self.tokenizer(self._raw_str, padding=True, return_tensors='pt')
        #print(encoded_inputs)


    

