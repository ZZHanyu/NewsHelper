# modules
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

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
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import gc

# AutoML SMAC: Auto Searching Hyperparameters
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    GreaterThanCondition,
    Float,
    InCondition,
    Integer,
)
from ConfigSpace.conditions import InCondition

from smac import HyperparameterOptimizationFacade, Scenario
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
from smac.runhistory.runhistory import RunHistory


# from other packages
from utils import preprocess
from utils import model
from main_class import main



class diagram_drawer():
    '''
        class used for draw diagram
    '''
    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots()
        self.__graph = None
        self.__batch_id = []
        self.__accuary = []
        self.flag = False

    
    def init_diagram(self, batch_id: int, accuary: float): 
        self.__batch_id.append(batch_id)
        self.__accuary.append(accuary)
        self.__graph = self.ax.plot(self.__batch_id, self.__accuary, color= 'g')[0]
        plt.xlabel("Batch id")
        plt.ylabel("Accurary")
        self.flag = True
        plt.show()


    def update_diagram(self, batch_id: int, accuary: float):
        if self.__graph == None:
            print("\n ** ERROR! no graph work found! \n")
            raise ModuleNotFoundError        
        self.__batch_id.append(batch_id)
        self.__accuary.append(accuary)
        self.__graph.set_xdata(self.__batch_id)
        self.__graph.set_ydata(self.__accuary)
        plt.show()









class configs(preprocess.data_handler):
    '''
        Auto ML auto Hyperparameters config:
            - learn rate
            - batch size
            - epoch
            - optimizer
            - activation function
            - Number/size of Hidden layer and Nerous
            -
    '''
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # 1 - Hyper-Parameter
            # Hyper-Value
        hidden_size = Integer("hidden_size", (16,512), default=128)
        hidden_size_linear = Integer("hidden_size_linear", (16,256), default=64)

        dropout_linear = Float("dropout_linear", (0.0, 0.5), default=0)

        activation_linear = Categorical("activation", ["tanh", "relu", "LeakyReLU"], default="LeakyReLU")

        num_layers = Integer("num_layers", (1,8), default=2)
        dropout = Float("LSTM_dropout", (0.0, 0.5), default=0)
        learn_rate = Float("Learning_Rate", (1e-5,1e-1), default=1e-3)

        batch_size = Integer("batch_size", (8, 256), default=32)
            
            # Choose optimizer method
        optimize_method = Categorical("optimizer", ["Adam", "SGD", "Adagrad", "RMSprop"], default="Adam")
        loss = Categorical("Loss_Function", ["BCEWithLogitsLoss", "MSELoss"], default="BCEWithLogitsLoss")

        
        # 2 - HyperParameter's Condition
        use_drop_out = GreaterThanCondition(child=dropout,
                                   parent=num_layers,
                                   value=1)
        use_drop_out_linear = GreaterThanCondition(child=dropout_linear,
                                          parent=num_layers,
                                          value=1)

       
        # Add hyperparameters and conditions to our configspace
        cs.add_hyperparameters(
            [hidden_size, 
             hidden_size_linear, 
             num_layers, 
             dropout, 
             dropout_linear, 
             learn_rate, 
             optimize_method, 
             loss, 
             activation_linear,
             batch_size]
        )
        
        # add conditions to configspace
        cs.add_conditions(
            [use_drop_out, 
             use_drop_out_linear]
        )

        return cs
    













class trainer(preprocess.data_handler):
    '''
        Our main train method
    '''
    def __init__(self) -> None:
        print("\n Now inital trainer..")        
        self.model = None
        self.optimizer = None
        self.loss = None
        self.normalize = None
        self._best_model_state = None
        self._flag = True
        self._result_list = []
        self.best_accurary = 0  
        self.config = configs()    
        self.diagram_drawer = diagram_drawer()
        # self.smac_model = True
        self.save_folder = main._args.model_save_path + main._args.date_time


        print("\nTrainer inital succefully!\n")       


    def display_all_params(self):
        print("\n\n *** starting print param:\n")
        for name, parms in self.model.named_parameters():
            logging.info(f"-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)} \n")
            print(f"-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)} \n")


    def save_model(self, save_path = None):
        self._best_model_state = copy.deepcopy(self.model.state_dict())
        if save_path != None:
            torch.save(self._best_model_state, f'{main._args.model_save_path}/{main._args.date_time}.pth')
        else:    
            torch.save(self._best_model_state, f'{save_path}/{main._args.date_time}model.pth')
        logging.info(f"\n\tModel save sucessful!\n")



    def _mini_batch(self, batch: list)-> None:

        target_set = []
        feature = []
        batch_size = len(batch)

        for data_idx in tqdm(range(batch_size), desc="MiniBatch Train", leave=False):
            feature.append(torch.tensor(batch[data_idx][0], dtype=torch.float32, device=main._device, requires_grad=True))
            target = torch.tensor([batch[data_idx][1]], dtype=torch.float32, device=main._device, requires_grad=True)
            target_set.append(target)
        seq_len = [s.size(0) for s in feature]
        pad_datas = pad_sequence(sequences=feature, padding_value=0.0, batch_first=False)
        pad_datas = pack_padded_sequence(input=pad_datas, lengths=seq_len, enforce_sorted=False, batch_first=False)

        # feature = torch.tensor(feature, requires_grad=True).to(main._device) # 513: 句子长度不一致 54个词/句子 vs 888个词/句子
        target_set = torch.tensor(target_set, dtype=torch.float32, device=main._device, requires_grad=True)
        

        y_pred = self.model.forward_with_batch(tensor_data=pad_datas, # feature
                                                batch_size=batch_size)
        y_pred_reshape = y_pred.reshape(-1)
        self.optimizer.zero_grad() # clean all grad
        loss = self.loss(y_pred_reshape, target_set)
        loss.backward()
        self.optimizer.step()
        
        return loss
    


    def _single_step(self, batch:list) -> float:

        total_loss = 0

        for data_idx in tqdm(range(len(batch)), desc="Single Step Train", leave=False):
            feature = torch.tensor(batch[data_idx][0], dtype=torch.float32, device=main._device, requires_grad=True)
            target = torch.tensor([batch[data_idx][1]], dtype=torch.float32, device=main._device, requires_grad=True)
            y_pred = self.model.forward(feature)
            self.optimizer.zero_grad() # clean all grad
            loss = self.loss(y_pred, target)
            loss.backward()
            total_loss += loss
            self.optimizer.step()
               
        total_loss /= len(batch)    
        logging.info(f" * Batch size = {len(batch)} , avg_loss = {total_loss}%\n")
        
        return total_loss
        


    
    
    def init_train(self, config: Configuration, seed: int = 0, budget: int = 25):
        '''
            Used for selecting a specfic hyperparameter
        '''
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if self.model == None or self.optimizer == None or self.loss == None:
                config_dict = config.get_dictionary()
                
                self.normalize = nn.BatchNorm1d(num_features=8).to(main._device)

                learn_rate = config_dict['Learning_Rate']

                self.model = LstmNet(
                        hidden_size=config_dict['hidden_size'],
                        hidden_size_linear=config_dict['hidden_size_linear'],
                        num_layers=config_dict['num_layers'],
                        dropout=config_dict['LSTM_dropout'],
                        dropout_linear=config_dict['dropout_linear'],
                        activation_linear=config['activation']
                    ).to(main._device)

                # optimize method choose
                match config_dict['optimizer']:
                    case 'Adam':
                        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learn_rate)
                    case 'SGD':
                        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=learn_rate)
                    case 'Adagrad':
                        self.optimizer = torch.optim.Adagrad(params=self.model.parameters(), lr=learn_rate)
                    case 'RMSprop':
                        self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=learn_rate)
                    case _:
                        raise ModuleNotFoundError
            

                # loss function choose
                match config_dict['Loss_Function']:
                    case 'BCEWithLogitsLoss':
                        self.loss = nn.BCEWithLogitsLoss()
                    case 'MSELoss':
                        self.loss = nn.MSELoss()
                    case _:
                        raise ModuleNotFoundError
                    
        cost = self.train(batch_size = config['batch_size'])
        return cost
    


    def manual_init_train(self):
        '''
            To init train config dict here
        '''

        # 使用random函数随机选择
        
        param_dict = {
            "hidden_size" : [32, 64, 128, 256],
            "hidden_size_linear" : [32, 64, 128, 256],
            "num_layers" : [1, 2, 3, 4, 5, 6, 7, 8],
            "dropout_linear" : [0.1, 0.2, 0.3, 0.4, 0.5],
            "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
            "activation_linear" : ["tanh", "relu", "LeakyReLU"],
            "learn_rate": [1e-5, 1e-4, 1e-3, 1e-2],
            "batch_size": [8 ,16, 24, 32, 48, 64],
            "optimize_method": ["Adam", "SGD", "Adagrad", "RMSprop"],
            "loss": ["BCEWithLogitsLoss", "MSELoss"]
        }


        for epoch_id in tqdm(range(self._args.num_epoches), desc="Epoch No.", leave=True):
            param_selected = {
                "hidden_size": None,
                "hidden_size_linear": None,
                "num_layers": None,
                "dropout_linear": None,
                "activation_linear": None,
                "dropout": None,
                "learn_rate": None,
                "batch_size": None,
                "optimize_method": None,
                "loss": None,
                "train_loss": None,
                "test_loss":None
            }


            preprocess.data_handler.initialize(batch_size=param_selected['batch_size'])

            print(f"\n --------epoch:{epoch_id} :param random choice--------\n")
            logging.info(f"\n --------epoch:{epoch_id} : param random choice--------\n")
            for key, item in param_dict.items():
                if param_selected.get(key, False):
                    raise ValueError
                else:
                    if key == "batch_size":
                        param_selected["batch_size"] = 2
                        continue
                    param_selected[key] = random.choice(item)
                    print(f"param : {key} --- value: {param_selected[key]}")
                    logging.info(f"param : {key} --- value: {param_selected[key]}")


            self.normalize = nn.BatchNorm1d(num_features=8).to(main._device)
            learn_rate = param_selected['learn_rate']

            self.model = LstmNet(
                hidden_size=param_selected['hidden_size'],
                hidden_size_linear=param_selected['hidden_size_linear'],
                num_layers=param_selected['num_layers'],
                dropout=param_selected['dropout'],
                dropout_linear=param_selected['dropout_linear'],
                activation_linear=param_selected['activation_linear']
                ).to(main._device)


            match param_selected['optimize_method']:
                case 'Adam':
                    self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learn_rate)
                case 'SGD':
                    self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=learn_rate)
                case 'Adagrad':
                    self.optimizer = torch.optim.Adagrad(params=self.model.parameters(), lr=learn_rate)
                case 'RMSprop':
                    self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=learn_rate)
                case _:
                    raise ModuleNotFoundError
                

            # loss function choose
            match param_selected['loss']:
                case 'BCEWithLogitsLoss':
                    self.loss = nn.BCEWithLogitsLoss()
                case 'MSELoss':
                    self.loss = nn.MSELoss() 
                case _:
                    raise ModuleNotFoundError
            

            # made a folder to store model .pth and config parameter
            new_folder_path = f'{self.save_folder}_{epoch_id}'
            os.mkdir(path=new_folder_path)
            with open(f'{new_folder_path}/config.json', 'w+') as f:
                json.dump(param_selected, f)


            logging.info(f"\n**** inital done, now start training ....\n")
            
            try:
                avg_cost = self.train(batch_size = param_selected['batch_size'], new_folder_path=new_folder_path)
                logging.info(f"\n avg_cost = {avg_cost}!\n")
            except Exception as e:
                print(f"\n ERROR ON Training: {e} \n")
                logging.info(f"\n ERROR ON Training: {e} \n")
                self.save_model(new_folder_path)
                logging.info(f"\n log memory error summary: {torch.cuda.memory_summary()}\n")
                continue


            # param_selected['train_loss'] = avg_cost
            logging.info(f"\n**** train finished! now testing ....\n")

            # now test
            total_loss = self.test(batch_size= param_selected['batch_size'])
            logging.info(f"\n ** total_loss = {total_loss}!\n")
            # param_selected['test_loss'] = total_loss.cpu().numpy()
            # record testing result
            try:
                json_result = json.dumps(param_selected)
            except Exception:
                json_result = None
                pass
            
            if json_result != None:
                with open('data.json', 'w') as f:
                    json.dump(json_result, f, indent=4)
                    logging.info(f"\n ***** epoch : {epoch_id} result save to json done! \n")
                logging.info(f"\n epoch:{epoch_id} : Single epoch finished! \n\n\n\n\n\n")
            else:
                logging.info(f"\nERROR! some error occur in json saver, now pass this part...\n")
                pass
            




    def train(self, batch_size, new_folder_path)-> float:    
        '''
            Our train method:
            1. SGD: every step optimized
            2. mini-batch: after whole batch then updated
        '''

        single_epoch_bar = tqdm(total=preprocess.data_handler._csv_total_len, desc="inside single epoch", leave=False)        
        self.model.train()

        data_generator = super().get_generator(batch_size = batch_size)
        avg_cost = 0
        count = 0  
        
        # for every epoch, reset the data genertor
        preprocess.data_handler.reset(batch_size)

        # Version 2: Using data generator to handle raw data only when trainer need them 
        try:
            while True:
                # a cuda memory release module:
                if main._device == torch.device("cuda"):
                    gc.collect()
                    torch.cuda.empty_cache()

                single_epoch_bar.update(batch_size)
                single_chunk_tuple = next(data_generator, None)
                single_chunk_idx = single_chunk_tuple[0]
                single_chunk = single_chunk_tuple[1]
                # epoch error, stop and save
                if not isinstance(single_chunk, list):
                    break
                match main._args.batch_model:
                    case True:
                        batch_loss = self._mini_batch(batch=single_chunk)
                        avg_cost += (1 - batch_loss)
                        print(f"\n batch loss = {batch_loss}\n")
                        logging.info(f"\t --> chunk_id = {single_chunk_idx} -> batch_loss = {batch_loss}")
                    case False:
                        step_loss = self._single_step(batch=single_chunk)
                        avg_cost += (1 - step_loss)
                        print(f"\n single loss = {step_loss}\n")
                        logging.info(f"\t --> chunk_id = {single_chunk_idx} -> batch_loss = {step_loss}")
                    case _:
                        raise KeyError
                
                # with torch.no_grad():
                #     if not self.diagram_drawer.flag:
                #         # need init draw helper
                #         self.diagram_drawer.init_diagram(batch_id=single_chunk_idx, accuary=step_loss.detach().cpu().numpy())
                #     else:
                #         self.diagram_drawer.update_diagram(batch_id=single_chunk_idx, accuary=step_loss.detach().cpu().numpy())
                
                count += 1
        except Exception as e:
            print(f"\n -- Trainer Error! error is = \t{e}\n")
            self.save_model(new_folder_path)
        

        print("\n** 1 epoch Done, Now SAVEING model! **\n")        
        self.save_model(new_folder_path)
        print("\nMODEL SAVED!\n")

        avg_cost = avg_cost / count
        print(f"\n\t **** AVG_Cost in 1 epoch = {avg_cost} ***** \n")
        logging.info(f"\n\t **** AVG_Cost in 1 epoch = {avg_cost} ***** \n")
        single_epoch_bar.close()
        
        return avg_cost
    

    def test(self, batch_size):
        self.model.eval()
        total_sample = 0
        test_true = 0 

        data_generator = super().get_generator(batch_size = batch_size)

        total_loss = 0

        for _ in range(20):
            single_chunk_tuple = next(data_generator, None)
            single_chunk_idx = single_chunk_tuple[0]
            single_chunk = single_chunk_tuple[1]
            total_sample += len(single_chunk)
            total_loss = 0
            
            for data_idx in tqdm(range(len(single_chunk)), desc="Single Step Train", leave=False):
                feature = torch.tensor(single_chunk[data_idx][0], dtype=torch.float32, device=main._device, requires_grad=True)
                target = torch.tensor([single_chunk[data_idx][1]], dtype=torch.float32, device=main._device, requires_grad=True)
                y_pred = self.model.forward(feature)
                
                if y_pred >= 0.5 and target[0] == 1:
                    test_true += 1
                elif y_pred <= 0.5 and target[0] == 0:
                    test_true += 1
                else:
                    continue
                
                loss = self.loss(y_pred, target)
                total_loss += loss
                
        total_loss /= 20
        total_accuary = test_true / total_sample
        print(f" \n\t ** Test message: accuary -> {total_accuary * 100}% ---> avg_loss (test) -> {total_loss}! ** \n")
        logging.info(f" \n\t ** Test Result: accuary -> {total_accuary * 100}% ---> avg_loss (test) -> {total_loss}! ** \n")

        return total_loss
        {
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
            #     self.save_model()
            #     print(f"\n * ERROR {e}! But model have been saved! \n")
        }
        


    def auto_ml(self):
        '''
            define parameter selection strategy
        '''

        facades: list[AbstractFacade] = []
        for intensifier_object in [SuccessiveHalving, Hyperband]: # 两种不同的优化策略
            '''
            SuccessiveHalving : 实施连续减半，支持多保真度、多目标和多处理。此增强器的行为如下：-首先，将运行历史的配置添加到跟踪器中。

            '''
            scenario = Scenario(
                self.config.configspace,
                walltime_limit=60,  # After 60 seconds, we stop the hyperparameter optimization
                n_trials=500,  # Evaluate max 500 different trials
                min_budget=1,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
                max_budget=25,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
                n_workers=1,
            )

            print(f"\n * scenario = {scenario} \n")
            
            # 在开始训练前fetch 5个 random config parameters
            initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

            # Create our intensifier
            intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")

            smac = MFFacade(
                scenario,
                self.init_train,
                initial_design = initial_design,
                intensifier = intensifier,
                overwrite=True,
            )

            print(f"\n * smac = {smac}\n")
            incumbent = smac.optimize()

            # run history display
            print(f"\n *** here is the run history = {smac.runhistory}\n")

            print(f"\n * incumbent = {incumbent}\n")
            facades.append(smac)

        




class LstmNet(nn.Module, model.module):
    '''
        Network:
            - LSTM + Double-Linear
    '''
    
    def __init__(self,
                 hidden_size,
                 hidden_size_linear,
                 num_layers,
                 dropout,
                 dropout_linear,
                 activation_linear):
        
        super(LstmNet, self).__init__()    
        model.module.__init__(self)


        '''
            Construct network layer:
            two options: 
                1, use smac to find best parameter (BETA) 
                2, choose pre-defined parameters
        '''

        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            bidirectional=True,
                            dropout=dropout)
        
        activation = None
        match activation_linear:
            case 'tanh':
                activation = nn.Tanh()
            case 'relu':
                activation = nn.ReLU()
            case 'LeakyReLU':
                activation = nn.LeakyReLU()
            case _:
                raise ModuleNotFoundError
        
        # construct linear layer (MLP)
        self.linears = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size_linear), # [lstm hidden dim, num class]
            activation,
            nn.Dropout(dropout_linear),
            nn.Linear(hidden_size_linear, 1) # [hidden dim, num class]
        )


        # init weight
        self.init_weights()


        self.hidden_size = hidden_size
        self.num_layers = num_layers

        logging.info(f"\n --> Model weight initalization succefuly!\n")

    
    def __call__(self) -> None:
        self.display_model_info()


    def init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if param.dim() > 2:
                if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                        torch.nn.init.xavier_uniform_(param, 0.0)

        for name, param in self.linears.named_parameters():
            if param.dim() > 2:     
                if 'weight' in name:   
                    torch.nn.init.xavier_uniform(param)
                elif 'bias' in name:
                    torch.nn.init.xavier_uniform_(param, 0.0)
            
        
        logging.info("initize model parameter done!\n")


    def display_model_info(self):
        print(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")
        logging.info(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")
        for name, child in self.lstm.named_children():
            logging.info(name, child) 
        for name, module in self.lstm.named_modules():
            logging.info(name, module)
        for name, param in self.lstm.named_parameters():
            logging.info(name, param)
        
        print(f"\n")
    

    def _load_pretrained_embedding_weight(self):
        model = gensim.models.KeyedVectors.load_word2vec_format(main._args)
        weights = torch.FloatTensor(model.vectors) # formerly syn0, which is soon deprecated
        return weights
    
    def forward(self, tensor_data:torch.tensor):
        # inital parameters
        # 4 = 2 if bidirectional=True else 1, * num_layers, 
        h0 = torch.randn(2*self.num_layers, self.hidden_size, device=main._device, dtype=torch.float32)
        c0 = torch.randn(2*self.num_layers, self.hidden_size, device=main._device, dtype=torch.float32)
        
        # Normalization:
        # tensor_data = self.normalization(tensor_data)
        
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
        h0 = torch.randn(2*self.num_layers, batch_size , self.hidden_size, device=main._device, dtype=torch.float32)
        c0 = torch.randn(2*self.num_layers, batch_size , self.hidden_size, device=main._device, dtype=torch.float32)
        h, _ = self.lstm(tensor_data, (h0, c0))
        h, _ = pad_packed_sequence(h)
        pred = F.sigmoid(self.linears(h[-1, :]))
        return pred
        




        
