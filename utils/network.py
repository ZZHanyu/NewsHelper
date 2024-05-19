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

# AutoML SMAC: Auto Searching Hyperparameters
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from ConfigSpace.conditions import InCondition
from smac import HyperparameterOptimizationFacade, Scenario

# from other packages
from utils import preprocess
from utils import model
from main_class import main


'''
Hyperparameters:
- learn rate
- batch size
- epoch
- optimizer
- activation function
- Number/size of Hidden layer and Nerous
- 
'''


class configs():
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # 1 - Hyper-Parameter
            # Hyper-Value
        hidden_size = Integer("hidden_size", (16,512), default=128)
        num_layers = Integer("num_layers", (1,8), default=2)
        dropout = Float("LSTM_dropout", (0.0, 0.5), default=0.2)
        learn_rate = Float("Learning_Rate", (1e-5,1e-1), default=1e-3)
            # Choose optimizer method
        optimize_method = Categorical("optimizer", ["Adam", "SGD", "Adagrad", "Adadelta", "RMSprop"], default="Adam")
        loss = Categorical("Loss_Function", ["BCEWithLogitsLoss", "MSELoss"], default="BCEWithLogitsLoss")

        # 2 - Create dependencies
        # # 这些依赖关系的作用是确保在搜索超参数配置空间时，只有符合条件的组合才会被考虑，从而避免无效或不合理的配置。
        # # 这在超参数调优过程中非常重要，因为它可以减少搜索空间，提高调优效率，并确保生成的配置能够在实际训练中使用。
        
        # use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
		# 		# 这意味着 degree 这个超参数仅在 kernel 为 "poly" 时才有效。换句话说，只有在选择了多项式核函数时，才需要设置多项式的次数（即 degree 超参数）。

        # use_coef = InCondition(child=coef, parent=kernel, values=["poly", "sigmoid"])
        #    # 这意味着 coef 这个超参数仅在 kernel 为 "poly" 或 "sigmoid" 时才有效。
        #   # 换句话说，只有在选择了多项式核函数或 sigmoid 核函数时，才需要设置 coef 超参数。
        
        # use_gamma = InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"])
        #   # 这意味着 gamma 这个超参数仅在 kernel 为 "rbf"、 "poly" 或 "sigmoid" 时才有效。
        #   # 换句话说，只有在选择了径向基核函数、多项式核函数或 sigmoid 核函数时，才需要设置 gamma 超参数。
        
        # use_gamma_value = InCondition(child=gamma_value, parent=gamma, values=["value"])
		# 		# 这意味着 gamma_value 这个超参数仅在 gamma 为 "value" 时才有效。
		# 		# 换句话说，只有在选择了手动指定 gamma 值时，才需要设置 gamma_value 超参数。

        # Add hyperparameters and conditions to our configspace
        cs.add_hyperparameters([hidden_size, num_layers, dropout, learn_rate, optimize_method, loss])
        
        return cs










class trainer(preprocess.data_handler):
    def __init__(self) -> None:
        print("\n Now inital trainer..")        
        # self.model = LstmNet().to(main._device)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # self.loss = nn.BCEWithLogitsLoss()
        # self.normalize = nn.BatchNorm1d(num_features=8).to(main._device)
        # self._best_model_state = None
        # self._flag = True
        # self._result_list = []
        # self.best_accurary = 0      
        # self.config = configs()
        self.model = None
        self.optimizer = None
        self.loss = None
        self.normalize = None
        self._best_model_state = None
        self._flag = True
        self._result_list = []
        self.best_accurary = 0  
        self.config = configs()    


        print("\nTrainer inital succefully!\n")       

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # 1 - Hyper-Parameter
            # Hyper-Value
        hidden_size = Integer("hidden_size", (16,512), default=128)
        hidden_size_linear = Integer("hidden_size_linear", (16,256), default=64)

        num_layers = Integer("num_layers", (1,8), default=2)

        dropout = Float("LSTM_dropout", (0.0, 0.5), default=0.2)
        dropout_linear = Float("dropout_linear", (0.0, 0.5), default=0.2)

        activation_linear = Categorical("activation", ["tanh", "relu", "LeakyReLU"], default="LeakyReLU")

        learn_rate = Float("Learning_Rate", (1e-5,1e-1), default=1e-3)
            # Choose optimizer method
        optimize_method = Categorical("optimizer", ["Adam", "SGD", "Adagrad", "Adadelta", "RMSprop"], default="Adam")
        loss = Categorical("Loss_Function", ["BCEWithLogitsLoss", "MSELoss"], default="BCEWithLogitsLoss")

        # 2 - Create dependencies
        # # 这些依赖关系的作用是确保在搜索超参数配置空间时，只有符合条件的组合才会被考虑，从而避免无效或不合理的配置。
        # # 这在超参数调优过程中非常重要，因为它可以减少搜索空间，提高调优效率，并确保生成的配置能够在实际训练中使用。
        
        # use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
		# 		# 这意味着 degree 这个超参数仅在 kernel 为 "poly" 时才有效。换句话说，只有在选择了多项式核函数时，才需要设置多项式的次数（即 degree 超参数）。

        # use_coef = InCondition(child=coef, parent=kernel, values=["poly", "sigmoid"])
        #    # 这意味着 coef 这个超参数仅在 kernel 为 "poly" 或 "sigmoid" 时才有效。
        #   # 换句话说，只有在选择了多项式核函数或 sigmoid 核函数时，才需要设置 coef 超参数。
        
        # use_gamma = InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"])
        #   # 这意味着 gamma 这个超参数仅在 kernel 为 "rbf"、 "poly" 或 "sigmoid" 时才有效。
        #   # 换句话说，只有在选择了径向基核函数、多项式核函数或 sigmoid 核函数时，才需要设置 gamma 超参数。
        
        # use_gamma_value = InCondition(child=gamma_value, parent=gamma, values=["value"])
		# 		# 这意味着 gamma_value 这个超参数仅在 gamma 为 "value" 时才有效。
		# 		# 换句话说，只有在选择了手动指定 gamma 值时，才需要设置 gamma_value 超参数。

        # Add hyperparameters and conditions to our configspace
        cs.add_hyperparameters([hidden_size, num_layers, dropout, learn_rate, optimize_method, loss])
        
        return cs


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
            torch.save(self._best_model_state, f'{main._args.model_save_path}model{main._args.date_time}.pth')



    def force_save_model(self):
        self._best_model_state = copy.deepcopy(self.model.state_dict())
        torch.save(self._best_model_state, f'{main._args.model_save_path}model{main._args.date_time}.pth')
        logging.info("Force Saving Sucessful!\n")



    def _mini_batch(self, 
                    batch: list)-> None:
        
        logging.info("*** \t batch model = True ")
        target_set = []
        feature = []
        batch_size = len(batch)

        for data_idx in tqdm(range(batch_size), desc="MiniBatch Train", leave=False):
            feature.append(batch[data_idx][0])
            target = torch.tensor([batch[data_idx][1]], dtype=torch.float32, device=main._device, requires_grad=True)
            target_set.append(target)
        feature = torch.tensor(feature, requires_grad=True).to(main._device) # 513: 句子长度不一致 54个词/句子 vs 888个词/句子
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
            feature = torch.tensor(batch[data_idx][0], dtype=torch.float32, device=main._device, requires_grad=True)
            target = torch.tensor([batch[data_idx][1]], dtype=torch.float32, device=main._device, requires_grad=True)
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
        


    def train(self, config: Configuration, seed: int = 0):
        if self.model == None or self.optimizer == None or self.loss == None:
            config_dict = config.get_dictionary()
            
            self.normalize = nn.BatchNorm1d(num_features=8).to(main._device)


            learn_rate = config_dict['Learning_Rate']

            # optimize method choose
            match config_dict['optimize_method']:
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


            self.model = LstmNet(hidden_size=config_dict['hidden_size'],
                                hidden_size_linear=config_dict['hidden_size_linear'],
                                num_layers=config_dict['num_layers'],
                                dropout=config_dict['dropout'],
                                dropout_linear=config_dict['dropout_linear'],
                                activation_linear=config['activation_linear']
                                ).to(main._device)

                
        data_generator = super().get_generator()
        for epoch_idx in tqdm(range(main._args.num_epoches), desc="Epoch No.", leave=True):
            logging.info(f"----------------- Epoch: {epoch_idx} ----------------- \n")
            self.model.train()
            
            # Version 2: Using data generator to handle raw data only when trainer need them 
            try:
                while True:
                    single_chunk = next(data_generator, None)
                    # epoch stop
                    if single_chunk == None:
                        self.save_model()
                        break

                    match main._args.batch_model:
                        case True:
                            self._mini_batch(batch=single_chunk)
                        case False:
                            Accuary = self._single_step(batch=single_chunk)
                            cost = 1 - Accuary
                            
                        case _:
                            raise KeyError
            except Exception as e:
                print(f"\n{e}\n")
                self.force_save_model()
                break

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
            #     self.force_save_model()
            #     print(f"\n * ERROR {e}! But model have been saved! \n")
            }
                    
    def auto_ml(self):
        # SMAC Next, we create an object, holding general information about the run
        scenario = Scenario(
            self.config.configspace,
            walltime_limit=60,  # After 60 seconds, we stop the hyperparameter optimization
            n_trials=50,
            min_budget=1,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
            max_budget=25,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
            n_workers=8,
        )

        # we want to run the facade's default initial design, but we want to change the number
        # of initial configs to 5.
        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)

        # Create our intensifier
        intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")


        # Now we use SMAC to find the best hyperparameters
        smac = HyperparameterOptimizationFacade(
            scenario,
            self.train,
            initial_design=initial_design,
            overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
        )

        incumbent = smac.optimize()
        # Get cost of default configuration
        default_cost = smac.validate(self.configspace.get_default_configuration())
        logging.info(f"Default cost: {default_cost}")

        # Let's calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent)
        logging.info(f"Incumbent cost: {incumbent_cost}")
        

    def test(self, batch:list):
        self.model.eval()
        test_true = 0 
        test_result = 0

        totoal_lenght = len(batch)
        for single_data in batch:
            feature = torch.tensor(single_data[0], dtype=torch.float32, device=main._device, requires_grad=True)
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
    def __init__(self,
                 hidden_size,
                 hidden_size_linear,
                 num_layers,
                 dropout,
                 dropout_linear,
                 activation_linear):
        
        super(LstmNet, self).__init__()    
        model.module.__init__(self)
  
        # construct LSTM layer
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

        logging.info(f"\n --> Model weight initalization succefuly!\n")

    
    def __call__(self) -> None:
        self.display_model_info()


    def init_weights(self) -> None:
        torch.nn.init.xavier_uniform(self.lstm.weight)
        torch.nn.init.xavier_uniform(self.linears.weight)
        

    def display_model_info(self):
        print(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")
        logging.info(f"\n Model Initialization = {self.lstm}\n *Parameters = {self.lstm.parameters}\n")
        print(f"\n")
    

    def _load_pretrained_embedding_weight(self):
        model = gensim.models.KeyedVectors.load_word2vec_format(main._args)
        weights = torch.FloatTensor(model.vectors) # formerly syn0, which is soon deprecated
        return weights
    
    def forward(self, tensor_data:torch.tensor):
        # inital parameters
        # 4 = 2 if bidirectional=True else 1, * num_layers, 
        h0 = torch.randn(4, 128, device=main._device, dtype=torch.float32)
        c0 = torch.randn(4, 128, device=main._device, dtype=torch.float32)
        
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
        h0 = torch.randn(4, batch_size , 128, device=main._device, dtype=torch.float32)
        c0 = torch.randn(4, batch_size , 128, device=main._device, dtype=torch.float32)
        h, _ = self.lstm(tensor_data, (h0, c0))
        pred = F.sigmoid(self.linears(h[-1, : , :]))
        return pred
        




        
