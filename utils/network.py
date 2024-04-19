# modules
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.nn.init as init
import os
import json
import time
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime, timedelta

import gensim


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
        self.model = LstmNet(main_args).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss = nn.BCEWithLogitsLoss()
        self.normalize = nn.BatchNorm1d(num_features=8).to(device)
        self._test_data = []


    def _train(self, batch, idx:int):
        self.model.train()
        self.optimizer.zero_grad() # clean all grad
        Accurary = 0
        # # Every 2 epochs/batches, lower the learning rate
        # if idx % 3 == 0:
        #     for param in self.optimizer.param_groups:
        #         param['lr'] = param['lr'] * self.learn_rate_decay

        y_pred_set = []
        target_set = []
        for data_idx in tqdm(range(len(batch)), desc="Batch No. {}".format(idx)):
            # feature = torch.tensor(batch[data_idx][0], dtype=torch.float32, device=device, requires_grad=True)
            feature = batch[data_idx][0].to(device)
            # feature = self.normalize(feature)
            target = torch.tensor([batch[data_idx][1]], dtype=torch.float32, device=device, requires_grad=True)
            # target = batch[data_idx][1]
            y_pred = self.model.forward(feature)

            
            if y_pred >= 0.5 and target[0] == 1:
                Accurary += 1
            elif y_pred < 0.5 and target[0] == 0:
                Accurary += 1

            
            y_pred_set.append(y_pred)
            target_set.append(target)

            

            # print(f"\n RESULT = predicted = {y_pred} and target = {target}\n")
            # loss = self.loss(y_pred, target)
            # loss.backward()
            
            # for name, parms in self.model.named_parameters():
            #     # print(f"grad = {parms.grad} \n and type = {type(parms.grad)}\n")
            #     # print(f"-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)}")
            #     logging.info(f"-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)} \n")
            
            #self.optimizer.step()
        y_pred_set = torch.tensor(y_pred_set, requires_grad=True)
        target_set = torch.tensor(target_set, requires_grad=True)
        loss = self.loss(y_pred_set, target_set)
        loss.backward()
        self.optimizer.step()

        
            
        Accurary /= len(batch)    
        logging.info(f"\n ** Round {idx} : Batch size = {len(batch)} , Accurary = {Accurary * 100}%\n")
        print(f"\n ** Round {idx} : Batch size = {len(batch)} , Accurary = {Accurary * 100}%\n")



    def _test(self):
        self.model.eval()
        test_true = 0 
        totoal_lenght = 0

        for single_batch in tqdm(self._test_data, desc="TEST Dataset"):
            for single_data in single_batch:
                totoal_lenght += len(single_batch)
                feature = torch.tensor(single_data[0], dtype=torch.float32, device=device, requires_grad=True)
                # target = torch.tensor([single_data[1]], dtype=torch.float32, device=device, requires_grad=False)
                target = single_data[1]
                # if single_data[1] == 1:
                #     target = [0,1] 
                # else: 
                #     target = [1,0]
                # target = torch.tensor(target, dtype=torch.float32, device=device, requires_grad=True)

                # print(f"\nfeature --> {feature} \n --> {type(feature)} \n --> size  {feature.size()}")
                y_pred = self.model.forward(feature)
                logging.info(f"TEST Processing --> pred = {y_pred} target = {target}")
                # STEP activiate function:
                if y_pred.item() >= 0.5 and target == 1:
                    test_true += 1
                elif y_pred.item() < 0.5 and target == 0:
                    test_true += 1
        logging.info(f"\n** TEST RESULT --> Accurary = {(test_true / totoal_lenght) * 100}% **\n")
    
    def _save_model(self):
        for param_tensor in self.model.state_dict():
            logging.info(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        for var_name in self.optimizer.state_dict():
            logging.info(var_name, "\t", self.optimizer.state_dict()[var_name])
        
        try:
            torch.save(self.model.state_dict(), self.args.model_save_path)
        except Exception as e:
            print(f"\n ERROR MODEL SAVING! {e}\n")
            logging.info(f"\n** Model saving ERROR, {e}\n")
        logging.info(f"** Model Saving Sucessfully!\n")
        print("\n Model saved! \n")
        

    def start(self, batch: list, idx:int):
        #FOR TRAIN AND THEN TEST
        if idx <= self.args.max_epochs:
            self._train(batch, idx)
        elif (idx - self.args.max_epochs) <= self.args.test_batch:
            self._test_data.append(batch)
        else:
            self._save_model()
            self._test()
        
        
        # # FOR TEST ONLY
        # if idx <= self.args.test_batch:
        #     # start test:
        #     # self._test_data = torch.cat((self._test_data, batch), dim=0)
        #     self._test_data.append(batch)
        # else:
        #     self._test()
        

class LstmNet(nn.Module):
    def __init__(self, args) -> None:
        super(LstmNet, self).__init__()
        self.main_args = args


        # weight = self._load_pretrained_embedding_weight()
        #self.embedding = nn.Embedding.from_pretrained(weight, freeze=True)

        self.normalization = nn.BatchNorm1d(num_features=300)


        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=128, 
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.2)
        
        '''
            >>> rnn = nn.LSTM(10, 20, 2)
            >>> input = torch.randn(5, 3, 10)
            >>> h0 = torch.randn(2, 3, 20)
            >>> c0 = torch.randn(2, 3, 20)
            >>> output, (hn, cn) = rnn(input, (h0, c0))
        '''
        
        # self.linear = nn.Linear(256, 1)
        # self.dropout = nn.Dropout(0.2)

        self.linears = nn.Sequential(
            nn.Linear(256, 64), # [lstm hidden dim, num class]
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1) # [hidden dim, num class]
        )


        # inilize weight
        self.linears.apply(self.init_weights) 
    
        # self._max_epochs = args.max_epochs
        # self._threshold = 0.5
        print(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")
        logging.info(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用均匀分布初始化权重
            init.xavier_uniform_(m.weight)
            # 初始化偏置为0
            init.constant_(m.bias, 0.0)
    

    def _load_pretrained_embedding_weight(self):
        model = gensim.models.KeyedVectors.load_word2vec_format(self.main_args)
        weights = torch.FloatTensor(model.vectors) # formerly syn0, which is soon deprecated
        return weights
    
    def forward(self, tensor_data:torch.tensor):
        # print(f"Shape tensor data = {tensor_data.size()},\n truth_v = {truth_v.size()}\n")
        # print(f"\n input = {tensor_data}, \n label = {truth_v}\n")


        # inital parameters
        h0 = torch.randn(4, 128, device=device, dtype=torch.float32)
        c0 = torch.randn(4, 128, device=device, dtype=torch.float32)
        
        # forward:
        # print(type(tensor_data))
        
        # Normalization:
        # tensor_data = self.normalization(tensor_data)
        # print(f"\n After normalization = {tensor_data}")
        h, _ = self.lstm(tensor_data, (h0, c0))
        # h = F.relu(h)
        # in [sequence_lenght,  hidden_feature_dim]
        # out [sequence_lenght, hidden dim]
        
        # pred = F.relu(self.dropout(self.linear(h)))
        #pred = F.sigmoid(self.dropout(self.linear(h[-1, :]))) # tensor size = [1,256]
        # pred = F.sigmoid(self.linear(h[-1, :]))
        # pred = self.linear(h[-1, :])
        #print(h[-1, :])
        #print(h[-1, :].size())
        pred = F.sigmoid(self.linears(h[-1, :]))
        # print(pred)

        # pred = self.dropout(self.linear(h)) 
        # in [sequence_lenght, hidden dim]
        # out [hidden dim, output size]    
        # 输出的形状为 [sequence_length, 1] 的矩阵表示了每个时间步上的模型预测结果。
        # 在这种情况下，每个时间步的预测结果是一个标量值，表示该时间步上模型对样本属于正类的置信度或概率。

        return pred



        
