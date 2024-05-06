from main_class import main
from utils import preprocess
from utils import model
from utils import topicModel
from utils import network



super_class = main()
data_handler = preprocess.data_handler()
data_handler.run()
trainer = network.trainer()
trainer.train()
LSTM_net = network.LstmNet()


