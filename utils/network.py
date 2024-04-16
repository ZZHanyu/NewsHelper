# modules
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import os
import json
import time
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime, timedelta

# utils
# from classifier import main


# check avaliable devices
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    if torch.backends.mps.is_built():
        device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"\n Devices selected = {device} \n")


class trainer():
    def __init__(self, main_args) -> None:
        self.args = main_args
        self.learn_rate_decay = 0.9
        self.network = LstmNet(main_args).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        self.loss = nn.BCELoss()
        self.normalize = nn.BatchNorm1d(num_features=8).to(device)
        self._test_data = None


    def _train(self, batch, idx:int):
        batch_avg_loss = 0

        # every 2 epochs/ batches, clear all grad cache and lower the lr value
        if idx % 2 == 0:
            self.optimizer.zero_grad() # clean all grad to save memeory
            for param in self.optimizer.param_groups:
                param['lr'] = param['lr'] * self.learn_rate_decay
        
        for data_idx in tqdm(range(len(batch)), desc="Batch No. {}".format(idx)):
            feature = torch.tensor(batch[data_idx][0], dtype=torch.float32, device=device, requires_grad=True)
            # feature = self.normalize(feature)
            target = torch.tensor([batch[data_idx][1]], dtype=torch.float32, device=device, requires_grad=False)
            y_pred = self.network.forward(feature)
            loss = self.loss(y_pred, target) / 100
            loss.backward()
            self.optimizer.step()
            batch_avg_loss += loss.item()
        
        batch_avg_loss /= len(batch)    
        logging.info(f"\n ** Round {idx} : Batch size = {len(batch)} , avg loss = {batch_avg_loss}\n")
        print(f"\n ** Round {idx} : Batch size = {len(batch)} , avg loss = {batch_avg_loss}\n")
        for name, parms in self.network.named_parameters():
            # print(f"-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)}")
            logging.info(f"-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)} \n")

    
    def _test(self):
        self.network.eval()
        test_true = 0 
        totoal_lenght = 0
        for single_batch in tqdm(self._test_data, desc="TEST Dataset"):
            # print(single_batch)
            print(f"\nBatch is --> type {type(single_batch)} --> {len(single_batch)}\n --> {single_batch}")
            totoal_lenght += len(single_batch)
            feature = torch.tensor(single_batch[0], dtype=torch.float32, device=device, requires_grad=True)
            target = torch.tensor([single_batch[1]], dtype=torch.float32, device=device, requires_grad=False)
            # print(f"\nfeature --> {feature} \n --> {type(feature)} \n --> size  {feature.size()}")
            y_pred = self.network.forward(feature)
            logging.info(f"TEST Processing --> pred = {y_pred} target = {target[0]}")
            # STEP activiate function:
            if y_pred > 0.5 and target[0] == 1:
                test_true += 1
            elif y_pred < 0.5 and target[0] == 0:
                test_true += 1
        logging.info(f"\n** TEST RESULT --> Accurary = {(test_true / totoal_lenght) * 100}% **\n")
        

    def start(self, batch: list, idx:int):
        #FOR TRAIN AND THEN TEST
        if idx <= self.args.max_epochs:
            self._train(batch, idx)
        elif (idx - self.args.max_epochs) <= self.args.test_batch:
            # start test:
            if self._test_data == None:
                self._test_data = batch
            else:
                self._test_data.append(batch)
        else:
            self._test(self._test_data)
        
        
        # FOR TEST ONLY
        # if idx <= self.args.test_batch:
        #     # start test:
        #     if self._test_data == None:
        #         self._test_data = batch
        #     else:
        #         # self._test_data = torch.cat((self._test_data, batch), dim=0)
        #         self._test_data.append(batch)
        # else:
        #     self._test()
        

class LstmNet(nn.Module):
    def __init__(self, args) -> None:
        super(LstmNet, self).__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=128, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.1)
    
        self._max_epochs = args.max_epochs
        self._threshold = 0.5
        print(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")
        logging.info(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")

    
    def forward(self, tensor_data:torch.tensor):
        # print(f"Shape tensor data = {tensor_data.size()},\n truth_v = {truth_v.size()}\n")
        # print(f"\n input = {tensor_data}, \n label = {truth_v}\n")
        
        # forward:
        h, _ = self.lstm(tensor_data) 
        # in [sequence_lenght,  hidden_feature_dim]
        # out [sequence_lenght, hidden dim]
        
        # pred = F.relu(self.dropout(self.linear(h)))
        pred = F.sigmoid(self.dropout(self.linear(h)))
        # pred = self.dropout(self.linear(h)) 
        # in [sequence_lenght, hidden dim]
        # out [hidden dim, output size]    
        # 输出的形状为 [sequence_length, 1] 的矩阵表示了每个时间步上的模型预测结果。
        # 在这种情况下，每个时间步的预测结果是一个标量值，表示该时间步上模型对样本属于正类的置信度或概率。

        return pred[-1]



        
