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
import copy
import gensim
import math
import pandas as pd

# from other packages
from utils import preprocess
from utils import model
from main_class import main


class trainer(preprocess.data_handler):
    def __init__(self) -> None:
        print("\n Now inital trainer..")        
        super().__init__()
        self.model = LstmNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss = nn.BCEWithLogitsLoss()
        self.normalize = nn.BatchNorm1d(num_features=8).to(self.device)
        self._best_model_state = None
        self._flag = True
        self._result_list = []
        self.best_accurary = 0        
        print("\nTrainer inital succefully!\n")        


    def display_all_params(self):
        print("\n\n *** starting print param:\n")
        for name, parms in self.model.named_parameters():
            logging.info(f"-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)} \n")
            print(f"-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)} \n")


    def save_model(self):
        # for param_tensor in self.model.state_dict():
        #     logging.info(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        # for var_name in self.optimizer.state_dict():
        #     logging.info(var_name, "\t", self.optimizer.state_dict()[var_name])
        if self._flag:
            self._best_model_state = copy.deepcopy(self.model.state_dict())
        else:
            torch.save(self._best_model_state, f'{self.args.model_save_path}model{self.args.date_time}.pth')

    def force_save_model(self):
        self._best_model_state = copy.deepcopy(self.model.state_dict())
        torch.save(self._best_model_state, f'{self.args.model_save_path}model{self.args.date_time}.pth')
        logging.info("Force Saving Sucessful!\n")


    def _mini_batch(self, 
                    batch: list)-> None:
        
        logging.info("*** \t batch model = True ")
        target_set = []
        feature = []
        batch_size = len(batch)

        for data_idx in tqdm(range(batch_size), desc="MiniBatch Train", leave=False):
            feature.append(batch[data_idx][0])
            target = torch.tensor([batch[data_idx][1]], dtype=torch.float32, device=self.device, requires_grad=True)
            target_set.append(target)
        feature = torch.tensor(feature, requires_grad=True).to(self.device) # 513: 句子长度不一致 54个词/句子 vs 888个词/句子
        target_set = torch.tensor(target_set, requires_grad=True)
        y_pred = self.model.forward_with_batch(tensor_data=feature,
                                                batch_size=batch_size)
        self.optimizer.zero_grad() # clean all grad
        loss = self.loss(y_pred, target_set)
        loss.backward()
        self.optimizer.step()




    def _single_step(self, 
                     batch:list) -> float:
        
        logging.info("*** \t batch model = False")
        Accurary = 0

        for data_idx in tqdm(range(len(batch)), desc="SGD", leave= False):
            feature = torch.tensor(batch[data_idx][0], dtype=torch.float32, device=self.device, requires_grad=True)
            target = torch.tensor([batch[data_idx][1]], dtype=torch.float32, device=self.device, requires_grad=True)
            y_pred = self.model.forward(feature)
            self.optimizer.zero_grad() # clean all grad
            loss = self.loss(y_pred, target)
            loss.backward()
            self.optimizer.step()
            
            # calculate accuary rate
            if y_pred >= 0.5 and target[0] == 1:
                Accurary += 1
            elif y_pred < 0.5 and target[0] == 0:
                Accurary += 1
            
        if len(batch) > 0:
            Accurary /= len(batch)    
        else:
            pass
        logging.info(f" * Batch size = {len(batch)} , Accurary = {Accurary * 100}%\n")
        
        return Accurary
        
    
    def train(self):
        data_generator = self.get_generator()
        for epoch_idx in tqdm(range(self.args.num_epoches), desc="Epoch No.", leave=True):
            logging.info(f"----------------- Epoch: {epoch_idx} ----------------- \n")
            self.model.train()
            
            # Version 2: Using data generator to handle raw data only when trainer need them 
            try:
                while next(data_generator, None) != None:
                    single_chunk = next(data_generator, None)
                    match self.args.batch_model:
                        case True:
                            self._mini_batch(batch=single_chunk)
                        case False:
                            self._single_step(batch=single_chunk)
                        case _:
                            raise KeyError
            except Exception as e:
                self.force_save_model()
                print(f"\nERROR: ' {e} ' had been seen!\n")



            # version 1: using old-school function, which store whole tokenzied dataset
            #   Drawback: Memory cost is extrmely high!
            # try:
            #     for single_chunk_idx, single_chunk in enumerate(tqdm(self._tokenized_chunks, leave=True, desc=f"Chunk No.{single_chunk_idx}")):
            #         if single_chunk_idx > math.floor(self.args.train_persentage * len(self.chunk_number)):
            #             self.test()
            #         else:
            #             if single_chunk_idx > 100 and single_chunk_idx % 77 == 0:
            #                 self.display_all_params()
            #                 self.save_model()
            #             match self.args.batch_model:
            #                 case True:
            #                     self._mini_batch(batch=single_chunk, idx=single_chunk_idx)
            #                 case False:
            #                     self._single_step(batch=single_chunk, idx=single_chunk_idx)
            #                 case _:
            #                     raise KeyError
            # except Exception as e:
            #     self.force_save_model()
            #     print(f"\n * ERROR {e}! But model have been saved! \n")
           
                    
                

            

    def test(self, batch:list):
        self.model.eval()
        test_true = 0 
        test_result = 0

        totoal_lenght = len(batch)
        for single_data in batch:
            feature = torch.tensor(single_data[0], dtype=torch.float32, device=self.device, requires_grad=True)
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
        test_result = (test_true / totoal_lenght) * 100
        logging.info(f"\n** TEST RESULT --> Accurary = {test_result}")
        # if better accuary get, then save
        if test_result > self.best_accurary:
            self.best_accurary = test_result
            self.save_model()
        else:
            pass
        return test_result
        




class LstmNet(nn.Module, model.module):
    def __init__(self):
        super(LstmNet, self).__init__()    
        model.module.__init__(self)    
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=128, 
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.2)
        
        # self.linear = nn.Linear(256, 1)
        # self.dropout = nn.Dropout(0.2)

        self.linears = nn.Sequential(
            nn.Linear(256, 64), # [lstm hidden dim, num class]
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1) # [hidden dim, num class]
        )

        # initalize all weights
        #self.init_weights()       
        logging.info(f"\n --> Model weight initalization succefuly!\n")

    
    def __call__(self) -> None:
        self.display_model_info()


    def init_weights(self):
        torch.nn.init.xavier_uniform(self.lstm.weight)
        torch.nn.init.xavier_uniform(self.linears.weight)
        

    def display_model_info(self):
        print(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")
        logging.info(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")
        print(f"\n")
    

    def _load_pretrained_embedding_weight(self):
        model = gensim.models.KeyedVectors.load_word2vec_format(self.args)
        weights = torch.FloatTensor(model.vectors) # formerly syn0, which is soon deprecated
        return weights
    
    def forward(self, tensor_data:torch.tensor):
        # print(f"Shape tensor data = {tensor_data.size()},\n truth_v = {truth_v.size()}\n")
        # print(f"\n input = {tensor_data}, \n label = {truth_v}\n")


        # inital parameters
        h0 = torch.randn(4, 128, device=self.get_device(), dtype=torch.float32)
        c0 = torch.randn(4, 128, device=self.get_device(), dtype=torch.float32)
        
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
    
    def forward_with_batch(self, tensor_data: torch.tensor, batch_size: int):
        h0 = torch.randn(4, batch_size , 128, device=self.device, dtype=torch.float32)
        c0 = torch.randn(4, batch_size , 128, device=self.device, dtype=torch.float32)
        h, _ = self.lstm(tensor_data, (h0, c0))
        pred = F.sigmoid(self.linears(h[-1, : , :]))
        return pred
        




        
