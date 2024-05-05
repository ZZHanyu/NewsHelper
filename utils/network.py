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
logging.info(f"\n *** Devices selected = {device} ! \n")


class trainer():
    def __init__(self,
                main_args,
                device) -> None:
        
        print("\nNow inital trainer..\n")        
        self.args = main_args
        self.learn_rate_decay = 0.9
        self.model = LstmNet(main_args).to(device)
        # self.display_all_params()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss = nn.BCEWithLogitsLoss()
        self.normalize = nn.BatchNorm1d(num_features=8).to(device)
        self._test_data = []
        self._best_model_state = None
        self._flag = True
        self._result_list = []
        self.best_accurary = 0        
        print("\nTrainer inital succefully!\n")        



    def display_all_params(self):
        print("\n\n *** starting print param:\n")
        for param in self.model.parameters():
            print(f"\n --> param = {param} --> type = {type(param)} --> size = {param.size()}\n")


    def _mini_batch(self, 
                    batch: list, 
                    idx: int)-> float:
        
        logging.info("*** \t batch model = True ")
        target_set = []
        feature = []
        batch_size = len(batch)

        for data_idx in tqdm(range(batch_size), desc="Batch No. {}".format(idx)):
            feature.append(batch[data_idx][0])
            target = torch.tensor([batch[data_idx][1]], dtype=torch.float32, device=device, requires_grad=True)
            target_set.append(target)
        feature = torch.tensor(feature, requires_grad=True).to(device)
        target_set = torch.tensor(target_set, requires_grad=True)
        y_pred = self.model.forward_with_batch(tensor_data=feature,
                                                batch_size=batch_size)
        self.optimizer.zero_grad() # clean all grad
        loss = self.loss(y_pred, target_set)
        loss.backward()
        self.optimizer.step()



    def _single_step(self, batch, idx) -> float:
        logging.info("*** \t batch model = False")
        Accurary = 0
        # # Every 2 epochs/batches, lower the learning rate
        # if idx % 3 == 0:
        #     for param in self.optimizer.param_groups:
        #         param['lr'] = param['lr'] * self.learn_rate_decay
            
        # pass required :
        #   batch, idx
        for data_idx in tqdm(range(len(batch)), desc="Batch No. {}".format(idx), leave= True):
            feature = torch.tensor(batch[data_idx][0], dtype=torch.float32, device=device, requires_grad=True)
            # feature = batch[data_idx][0].to(device)
            #feature = torch.tensor(feature, requires_grad=True).to(device)
            # feature = self.normalize(feature)
            target = torch.tensor([batch[data_idx][1]], dtype=torch.float32, device=device, requires_grad=True)
            # target = batch[data_idx][1]
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
            # if data_idx % 13 == 0:
            #     for name, parms in self.model.named_parameters():
            #         # print(f"\n -->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {parms.grad}")
            #         logging.info(f"-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)} \n")
        
        # set checkpoint in case of break
        if idx % 100 == 0 and idx > 100:
            self.save_model()
            for name, parms in self.model.named_parameters():
                logging.info(f"-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)} \n")

        if len(batch) > 0:
            Accurary /= len(batch)    
        else:
            pass
        logging.info(f"\n ** Training Round {idx} : Batch size = {len(batch)} , Accurary = {Accurary * 100}%\n")
        print(f"\n ** Training Round {idx} : Batch size = {len(batch)} , Accurary = {Accurary * 100}%\n")
        


    
    def train(self, 
               batch:list, 
               idx:int):

        self.model.train()
        if self.args.batch_model == True:
            self._mini_batch(batch=batch, idx=idx)
        else:
            self._single_step(batch=batch, idx=idx)

            

    def test(self, batch:list):
        self.model.eval()
        test_true = 0 
        test_result = 0
        # totoal_lenght = 0

        # for single_batch in tqdm(self._test_data, desc="TEST Dataset"):
        #     for single_data in single_batch:
        #         totoal_lenght += len(single_batch)
        #         feature = torch.tensor(single_data[0], dtype=torch.float32, device=device, requires_grad=True)
        #         # target = torch.tensor([single_data[1]], dtype=torch.float32, device=device, requires_grad=False)
        #         target = single_data[1]
        #         # if single_data[1] == 1:
        #         #     target = [0,1] 
        #         # else: 
        #         #     target = [1,0]
        #         # target = torch.tensor(target, dtype=torch.float32, device=device, requires_grad=True)

        #         # print(f"\nfeature --> {feature} \n --> {type(feature)} \n --> size  {feature.size()}")
        #         y_pred = self.model.forward(feature)
        #         logging.info(f"TEST Processing --> pred = {y_pred} target = {target}")
        #         # STEP activiate function:
        #         if y_pred.item() >= 0.5 and target == 1:
        #             test_true += 1
        #         elif y_pred.item() < 0.5 and target == 0:
        #             test_true += 1

        totoal_lenght = len(batch)
        for single_data in batch:
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
        test_result = (test_true / totoal_lenght) * 100
        logging.info(f"\n** TEST RESULT --> Accurary = {test_result}")
        # if better accuary get, then save
        if test_result > self.best_accurary:
            self.best_accurary = test_result
            self.save_model()
        else:
            pass

        return test_result
    
    def save_model(self):
        # for param_tensor in self.model.state_dict():
        #     logging.info(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        # for var_name in self.optimizer.state_dict():
        #     logging.info(var_name, "\t", self.optimizer.state_dict()[var_name])
        if self._flag:
            self._best_model_state = copy.deepcopy(self.model.state_dict())
        else:
            torch.save(self._best_model_state, f'{self.args.model_save_path}model{self.args.date_time}.pth')

    def save_model(self, ):
        pass

    def force_save_model(self):
        self._best_model_state = copy.deepcopy(self.model.state_dict())
        torch.save(self._best_model_state, f'{self.args.model_save_path}model{self.args.date_time}.pth')
        logging.info("Force Saving Sucessful!\n")
        

    def start(self, 
              batch: list, 
              idx:int, 
              flag: bool,
              total_chunk):
        
        self._flag = flag
        try:
            if self._flag:
                self.train(batch, idx, total_chunk)
            else:
                self.save_model()
                self.test(batch)
        except Exception as e:
            logging.info(f" \n\t *** ERROR in train and test, now start force saving...\n")
            self.force_save_model()


        
        
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

        # self.normalization = nn.BatchNorm1d(num_features=300)


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

        # self._max_epochs = args.max_epochs
        self.display_model_info()


    def init_weights(self):
        torch.nn.init.xavier_uniform(self.lstm.weight)
        torch.nn.init.xavier_uniform(self.linears.weight)
        

    def display_model_info(self):
        print(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")
        logging.info(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")
        print(f"\n")
    

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
    
    def forward_with_batch(self, tensor_data: torch.tensor, batch_size: int):
        h0 = torch.randn(4, batch_size , 128, device=device, dtype=torch.float32)
        c0 = torch.randn(4, batch_size , 128, device=device, dtype=torch.float32)
        h, _ = self.lstm(tensor_data, (h0, c0))
        pred = F.sigmoid(self.linears(h[-1, : , :]))
        return pred
        




        
