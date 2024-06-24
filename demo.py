import os
import time
import json
import torch
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_corpus, common_dictionary
import pandas as pd
from tqdm import tqdm

from main_class import main
from utils import preprocess
from utils import model
from utils import topicModel
from utils import network
from api import openai_api






    








main.initialize()
preprocess.data_handler.initialize(batch_size=200)
device = main._device


# for LSTM classification model
config_path = './trained_model/2024_06_11_18:46_0'
model_path = './trained_model/2024_06_11_18:46.pth'
with open(f'{config_path}/config.json') as conf:
    config_dict = json.load(conf)
classify_model = network.LstmNet(
        hidden_size = config_dict['hidden_size'],
        hidden_size_linear = config_dict['hidden_size_linear'],
        num_layers = config_dict['num_layers'],
        dropout = config_dict['dropout'],
        dropout_linear = config_dict['dropout_linear'],
        activation_linear = config_dict['activation_linear']
    ).to(device)
check_point = torch.load(model_path, map_location=device)
classify_model.load_state_dict(check_point)



def data_loader():
    # dataset loader
    single_chunk = next(preprocess.data_handler._chunks, None)
    preprocess.data_handler._remove_empty_line(single_chunk)
    if isinstance(single_chunk, pd.DataFrame):
        return single_chunk
    else:
        return None



def get_classfiy(data):
    real_news_list = []
    for row in tqdm(data.itertuples(), desc="Handling single row in a chunk", leave=False):
        temp = preprocess.data_handler._string_handler(raw_text=str(row[2]) + row[3]), row[4]
        feature = torch.tensor(temp[0], dtype=torch.float32, device=main._device, requires_grad=True)
        target = torch.tensor([temp[1]], dtype=torch.float32, device=main._device, requires_grad=True)
        y_pred = classify_model.forward(feature)
        
        if y_pred >= 0.5 and target[0] == 1:
            real_news_list.append(row[2]+row[3])

        else:
            continue

    return real_news_list




def get_lda(real_news_list):
    # for LDA model
    lda_path = './LDA_Model/lda_model2024_06_18'
    lda_model = LdaModel.load(lda_path)

    topic_model_vector = []

    realnews_corpus = [(text, common_dictionary.doc2bow(text.split())) for text in real_news_list]
    for unseen_text in realnews_corpus:
        topic_model_vector.append((unseen_text[0], lda_model[unseen_text[1]]))
    # lda.update(realnews_corpus)
    print("\nRESULT:\n")
    for i in topic_model_vector:
        print(f"\n{i}\n")
    

    class_result = {
            0:[],
            1:[],
            2:[],
            3:[],
            4:[]
            }


    with open('./tp_result.json', 'w+') as f:
        for i in tqdm(topic_model_vector, desc="analysis topic of truth news...", leave=True):
            maxium = 0
            maxium_label = None
            # find maxium:
            
            for single_label in i[1]:
                if single_label[1] > maxium:
                    maxium = single_label[1]
                    maxium_label = single_label[0]
                else:
                    continue
            class_result[maxium_label].append(i[0])
        json.dump(class_result, f)





def get_topic():
    result = []
    with open('./tp_result.json', 'r') as f:
        tp_result = json.load(f)
        for topic_idx in range(2, 5):
            text = tp_result[str(topic_idx)][0]
            prompt = openai_api.get_prompt(text)
            result.append(openai_api.get_response(prompt))


    # show the topic names:
    for idx, single_topic in enumerate(result):
        print(f"\n\t *** For the topic id {idx}, the topic descripion is {single_topic}! *** \n")













if __name__ == "__main__":
    while True:
        single_batch_data = data_loader()
        if not isinstance(single_batch_data, pd.core.frame.DataFrame):
            break
        real_news_set = get_classfiy(single_batch_data)
        get_lda(real_news_set)
        get_topic()

        choice = input("whether continue? 1 - yes; 0 - no")
        match(choice):
            case '1':
                continue
            case '0':
                break
            case _:
                raise ValueError

    print("\nProgram END!\n")









