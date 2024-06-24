import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd

from utils import preprocess


class mini_preprocess(preprocess.data_handler):
    def __init__(self) -> None:
        path = "/Users/taotao/Documents/GitHub/FYP/data/fake-news-classification/WELFake_Dataset.csv"
        data = pd.read_csv(path)
        
