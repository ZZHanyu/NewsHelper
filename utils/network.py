# modules
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import json
import time

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
        # super().__init__(main_args)
        self.args = main_args
        self.network = LstmNet(main_args)
        pass
    
    def _data_reader(self):
        # check files
        file_list = os.listdir(self.args.result_path)

        for file_idx in range(len(file_list)):
            if not os.path.exists(self.args.result_path + "tokenized{}.json".format(file_idx)):
                assert 1==2, print(f"\n ** ERROR! The JSON does NOT exists!\n")
            else:
                with open(self.args.result_path + "tokenized{}.json".format(file_idx), 'r') as json_file:
                    try:
                        file_content = json_file.read()
                        data = json.loads(file_content)
                        # print(f"JSON file read succueed! len = {len(data)} \ntype = {type(data)}\n")
                        for key, value in data.items():
                            #print(f"Key = {key}, \n Value detail = for 0 {type(value[0])}\n size = {len(value[0])} \n")
                            #time.sleep(5)
                            #print(f"\n\n *** Value in feature matrix = \n {value[0]}\n")
                            try:    
                                my_tensor = torch.tensor(value[0]).to(device)
                                target = torch.tensor(value[1]).to(device)
                                print(f"Round {key}: input size = {my_tensor.size()}, and label = {target.size()}, ")
                                self.network.forward(my_tensor, target)
                            except Exception as e:
                                print(f"ERROR! {e}\n")
                        # print(f"JSON file read succueed! context = {data}\n")
                    except Exception as e:
                        print(f"\n ERROR! {e}\n")
                    
                    
            
    
            # json_path = self.args.result_path + "tokenized{}.json".format()
            
        

    def start(self):
        self._data_reader()

        

class LstmNet(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self._lstm = nn.Sequential(
            nn.LSTM(input_size=512, hidden_size=512, num_layers=2, bidirectional=False),
            nn.Linear(512, 2)
        ).to(device)
        self._lossfn = nn.BCELoss().to(device)
        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=1e-2)
        self._max_epochs = args.max_epochs


    
    def forward(self, tensor_data, truth_v):
        print(f"Model = {self._lstm}\n Parameters = {self._lstm.parameters}")
        
        lstm_output, _ = self._lstm(tensor_data, (torch.randn(2, 1, 512), torch.randn(2, 1, 512)))
        loss = self._lossfn(torch.sigmoid(lstm_output), truth_v)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        print(f"loss: {loss}\n")

        
