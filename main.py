from main_class import main
from utils import preprocess
from utils import model
from utils import topicModel
from utils import network

import multiprocessing
import logging
from datetime import datetime, timedelta
import time


# Start of whole program:
if __name__ == "__main__":
    start_time = time.time()
    logging.basicConfig(filename=f'./log/{datetime.now().strftime("%Y_%m_%d_%H:%M")}.log', level=logging.INFO)
    logging.info('\nStarted Logging...\n')


    multiprocessing.freeze_support()
    # abstract class init
    main.initialize()
    # preprocess.data_handler.initialize()


    # start training LSTM network (fake news classifier)
    trainer = network.trainer()
    if main._args.use_smac == True:
        trainer.auto_ml()
    else:
        choice = input("whether costmize parameters? \n1.use defualt parameters, \n2.randomly choose parameter \n3. use post config\n")
        match choice:
            case '1':
                print("\n use defualt parameters")
                trainer.default_init_train()

            case '2':
                print("\n randomly choose parameter")
                trainer.manual_init_train()
            case '3':
                print("\n use post config") 
                trainer.use_past_config()
            case _:
                quit(1)
    


    # # topic modeling
    # tp_model = topicModel.LDA_topic_model()
    # tp_model.forward()

    

    end_time = time.time()
    logging.info(f"\n ** Total time cost = {end_time - start_time} s! \n")
    logging.info('\n----------------------- End loging -----------------------\n')