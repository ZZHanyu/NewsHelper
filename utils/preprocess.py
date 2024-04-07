from transformers import PreTrainedTokenizerFast
import pandas
from tqdm import tqdm
import time 
import re
import json


class charactors_hander:
    def __init__(self, chunks: pandas.io.parsers.readers.TextFileReader) -> None:
        self._meta_data = chunks
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-uncased')
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self._json_result = {}
        
    # def _remove_empty(self, chunk, *args):
    #     for arg in args:
    #         chunk = chunk[len(chunk[arg]) != 0]


    def _split_single_word(self) -> list:
        return self._raw_str.split()
    
    def display_elements(self):
        return self._raw_str

    def run(self):
        # try:
        for chunk in tqdm(self._meta_data, desc="Layer1: Data Preprocess"):
            # Step1: check whether empty
                # if a blank dectected, then delete this line
            # self._remove_empty(chunk, 'title', 'text', 'label')

            # STEP 1: Remove empty line (Remove dirty data)
            chunk = chunk[(chunk['title'].notnull()) & (chunk['title'].dtypes == object)]
            
            chunk = chunk[(chunk['text'].apply(lambda x: True if (x != None) & (type(x) == str) else False))]
            chunk = chunk[(chunk['label'].notnull()) & (chunk['label'].dtypes == int)]

            for index, row in chunk.iterrows():
                #print(f"index = {index}\n context = {row}, \ntitle = {row['title']}\n type = {type(row['title'])},\n length={len(row['title'])} \n")
                            
                # STEP 2: Lower all charactors in string
                row['title'] = row['title'].lower()
                row['text'] = row['text'].lower()

                print(f"{len(row['text'])}\n")
                
                # split
                row['title'] = row['title'].split()
                row['text'] = row['text'].split()

                # tokenizer
                print(f"{row['text']} {len(row['text'])} {type(row['text'])}\n")
                encoder_title = self.tokenizer(row['title'], padding=True, return_tensors='pt')
                encoder_text = self.tokenizer(row['text'], padding=True,  return_tensors='pt')
                #print(f"token result = title = {encoder_title}\n text = {encoder_text}\n\n")
                #time.sleep(5)
                self._json_result[index] = (encoder_title, encoder_text, row['label'])
        print(self._json_result)
                #print(f"**After modify : \t title = {row['title']}, context = {row['text']}, label = {row['label']} \n\n")

        # except Exception as e:
        #     print(f"Data Preprocessing Failed! error is {e}")


        
        # if self._empty_judge() == False:
        #     assert 1==0, print("** ERROR! there are blank line exist")
        
        


        # split into single words set



        #encoded_inputs = self.tokenizer(self._raw_str, padding=True, return_tensors='pt')
        #print(encoded_inputs)


    

