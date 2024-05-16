import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import os
import re
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
import gensim

from main_class import main


class module(main):
    '''
        abstract
    '''
    

    def __init__(self):
        super().__init__()
        
    def get_device(self):
        return self.device


    def get_data_iterator(self): # API
        if self.data_generator != None:
            return self.data_generator



    def forward(self):
        '''
            abstract method
        '''
        pass


    def run(self):
        '''
            abstract method
        '''
        pass


    def train(self):
        '''
            abstract method
        '''
        pass

