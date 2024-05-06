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
    def __init__(self) -> None:
        super().__init__()
        # self.args = main_args
        # self.device = device
        # #self.tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-uncased')
        self.embedding_model = gensim.downloader.load(self.args.pretrained_embedding_model_name)

        pass

    def forward(self):
        pass


    def run(self):
        pass


    def train(self):
        pass

