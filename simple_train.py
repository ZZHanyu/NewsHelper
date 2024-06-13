import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from torch.utils.data import DataLoader


import opendatasets as od
import math
import pandas as pd
import os

from main_class import main


# class simple_trainer(main):
#     def __init__(self) -> None:
#         main.initialize()
#         if not os.path.exists(super()._args.dataset_path + super()._args.dataset_name + '/WELFake_Dataset.csv'):
#             print(f"The file path {super()._args.dataset_path + 'WELFake_Dataset.csv'} is missing! Now downloading...\n")
    
#     def test(self):
#     # kaggle datasets download -d saurabhshahane/fake-news-classification
#         od.download('https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification', data_dir=super()._args.dataset_path)
#         main._chunks = pd.read_csv(super()._args.dataset_path + super()._args.dataset_name + '/' +'WELFake_Dataset.csv', chunksize=batch_size)
#         else:
#             cls._chunks = pd.read_csv(super()._args.dataset_path + super()._args.dataset_name + '/' +'WELFake_Dataset.csv', chunksize=batch_size)
#             pass



import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, file_path, chunksize, transform=None):
        self.file_path = file_path
        self.chunksize = chunksize
        self.transform = transform
        self.chunks = pd.read_csv(self.file_path, chunksize=self.chunksize)
        self.data = self.preprocess_chunk(next(self.chunks))

        
    def preprocess_chunk(self, chunk):
        # 将非数值列转化为数值列，或者删除这些列
        for column in chunk.columns:
            if chunk[column].dtype == 'object':
                chunk[column] = pd.factorize(chunk[column])[0]
    
    def __len__(self):
        # Return an arbitrary large number to simulate the length of the dataset
        # Note: you may need to adjust this based on your actual data size
        return 10**6
    
    def __getitem__(self, idx):
        try:
            # Get the current chunk
            if idx >= len(self.data):
                self.data = self.preprocess_chunk(next(self.chunks))
                idx = 0
            sample = self.data.iloc[idx]
        except StopIteration:
            self.chunks = pd.read_csv(self.file_path, chunksize=self.chunksize)
            self.data = self.preprocess_chunk(next(self.chunks))
            sample = self.data.iloc[0]
        
        if self.transform:
            sample = self.transform(sample)
        
        # Ensure the sample is converted to numeric values only
        sample = sample.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return torch.tensor(sample.values, dtype=torch.float32)

# Example usage
file_path = '/Users/taotao/Documents/GitHub/FYP/data/fake-news-classification/WELFake_Dataset.csv'
chunksize = 100  # Adjust based on your memory capacity

dataset = CustomDataset(file_path, chunksize)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Iterate through the DataLoader
for batch in dataloader:
    # Your training code here
    print(batch)
    break  # Remove this break statement for actual training loop