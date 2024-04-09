import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# utils
from utils import preprocess
# from utils import custom_dataloader
from utils import network

# other modules
import opendatasets as od
import os
from datetime import datetime

class main:
    def __init__(self, main_args) -> None:
        self.args = main_args
        self._chunks = None
        self.formatted_time = datetime.now().strftime("%Y_%m_%d")

    def _dataloader(self):
        '''
            ---------- Part 1: Dataset Loader ----------
            Objective:
                - using batchs(chunks) to load csv line by line
            Params:
                - args: chunk size defined
        '''
        if not os.path.exists(self.args.dataset_path + self.args.dataset_name + '/WELFake_Dataset.csv'):
            print(f"The file path {self.args.dataset_path + 'WELFake_Dataset.csv'} is missing! Now downloading...\n")
            # kaggle datasets download -d saurabhshahane/fake-news-classification
            od.download('https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification', data_dir=self.args.dataset_path)
            
        try:
            self._chunks = pd.read_csv(self.args.dataset_path + self.args.dataset_name + '/' +'WELFake_Dataset.csv', chunksize=self.args.chunk_size)
        except Exception as e:
            print(f"Data Loading Failed! {e}")
    

    def _data_preprocess(self):
        '''
            ---------- Part 2: Text Preprocess ----------
            Objective:
                - clean the symbols(,.%) and space 
                - lowercase the letters
            Params:

        '''
        data_handler = preprocess.charactors_hander(chunks=self._chunks, main_args=self.args)
        data_handler.run()

    def _lstm_net(self):
        lstm = network.LstmNet(self.args)
        lstm.forward()

    def forward(self):
        self._dataloader()
        self._data_preprocess()
        # self._lstm_net()
        lstmNet = network.trainer(main_args=args)
        lstmNet.start()





# Start from here
parser = argparse.ArgumentParser(description="Parameters for Classifier")
parser.add_argument("--dataset_path", type=str, default="/Users/taotao/Documents/GitHub/FYP/data/", help="the path of your dataset")
parser.add_argument("--dataset_name", type=str, default='fake-news-classification', help="the dataset name from kaggle")
parser.add_argument("--chunk_size", type=int, default=10000, help="control how many lines read once")
parser.add_argument("--max_epochs", type=int, default=10000, help="epochs of training")
parser.add_argument("--train_persentage", type=float, default=0.8, help="dataset persentage used for training")
parser.add_argument("--result_path", type=str, default="result/", help="result output destnation file")
parser.add_argument("--date_time", type=str, default=datetime.now().strftime("%Y_%m_%d"), help="date_form_Y_M_D")
args = parser.parse_args()

main_progress = main(args)
main_progress.forward()


