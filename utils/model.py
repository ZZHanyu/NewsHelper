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


class module():
    def __init__(self,
                 main_args,
                 device,
                 ) -> None:
        self.args = main_args
        self.device = device
        #self.tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-uncased')
        self.embedding_model = gensim.downloader.load(main_args.pretrained_embedding_model_name)

        pass

    def forward():
        pass


    def run():
        pass


