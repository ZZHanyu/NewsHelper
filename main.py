from main_class import main
from utils import preprocess
from utils import model
from utils import topicModel
from utils import network

import logging



#super_class = main()
#data_handler = preprocess.data_handler()


# sub-class model 
trainer = network.trainer()
trainer.train()


