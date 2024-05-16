from main_class import main
from utils import preprocess
from utils import model
from utils import topicModel
from utils import network




# abstract class init
main.initialize()
preprocess.data_handler.initialize()

# start training
trainer = network.trainer()
trainer.train()


