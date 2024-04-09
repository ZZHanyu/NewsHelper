# modules
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os

# utils
# from classifier import main
import json

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
        pass
    
    def _data_reader(self):
        # check files
        json_path = self.args.result_path + "tokenized" + self.args.date_time
        if not os.path.exists(json_path):
            assert 1==2, print(f"ERROR! The JSON does NOT exists!\n")
        else:
            with open(json_path, 'r') as json_file:
                for line in json_file:
                    data = json.loads(line)
                    print(data)

    def start(self):
        self._data_reader()
        



class LstmNet(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self._lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=2, bidirectional=True)
        self._lossfn = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=1e-2)
        self._max_epochs = args.max_epochs


    
    def forward(self):
        print(f"Model = {self._lstm}\n Parameters = {self._lstm.parameters}")
        
        
