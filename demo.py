from main_class import main
from utils import preprocess
from utils import model
from utils import topicModel
from utils import network

if __name__ == "__main__":
    config_path = '/root/autodl-tmp/NewsHelper/trained_model/2024_06_11_18:46_0'
    model_path = '/root/autodl-tmp/NewsHelper/trained_model/2024_06_11_18:46.pth'

    tp_model = topicModel.LDA_topic_model()
    tp_model.forward()
