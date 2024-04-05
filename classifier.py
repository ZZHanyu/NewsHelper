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






'''
    Part 1: Load Dataset
'''
def dataloader(args):
    try:
        chunks = pd.read_csv(args.dataset_path, chunksize=args.chunk_size)
        for chunk in chunks:
            # print(chunk)      
            for index, row in chunk.iterrows():
                print(f"{index}\t {row} \n")          
    except Exception as e:
        print(f"Data Loading Failed! {e}")
    
    




'''
    Part 2: Text Preprocess
    - clean the symbols(,.%) and space 
    - lowercase the letters
    - 
    
'''








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for Classifier")
    parser.add_argument("--dataset_path", type=str, default="utils/WELFake_Dataset.csv", help="the path of your dataset")
    parser.add_argument("--chunk_size", type=int, default=10000, help="control how many lines read once")

    args = parser.parse_args()

    #mydataset = custom_dataloader.CustomDataset()
    dataloader(args=args)


