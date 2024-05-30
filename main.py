from main_class import main
from utils import preprocess
from utils import model
from utils import topicModel
from utils import network

import multiprocessing


# Start of whole program:
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # abstract class init
    main.initialize()
    preprocess.data_handler.initialize()

    # start training
    trainer = network.trainer()
    trainer.auto_ml()


    # # topic modeling
    # tp_model = topicModel.LDA_topic_model()
    # tp_model.forward()