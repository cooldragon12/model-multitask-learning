from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer
import numpy as np

import matplotlib.pyplot as plt
# from utils.config import (
#     accuracy_emotion,
#     accuracy_toxicity,
#     precision_emotion,
#     precision_toxicity,
#     recall_emotion,
#     recall_toxicity,
#     f1_emotion,
#     f1_toxicity,
#     confusion_matrix_emotion,
#     confusion_matrix_toxicity,
#     AUROC_emotion,
#     AUROC_toxicity,
# )
from .config import BASE_PATH, EMOTIONS_LABELS, TOXICITY_LABELS




def show_loss_graph(loss, val_losses, title: str):
    """
    Shows the graph of the losses, save in photo
    """

    plt.figure()
    plt.plot(loss, label="train Loss")
    plt.plot(val_losses, label="val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training /Validation Loss")
    # Filename <datetime>_loss_graph.png
    plt.savefig(
        f'{BASE_PATH}/logs/losses/{title}_{datetime.now().strftime("%Y%m%d%H%M%S")}_loss_graph.png'
    )


def show_toxicity_loss_graph(toxicity_loss,toxicity_val_loss, title: str):
    """
    Shows the graph of the losses, save in photo
    """
    plt.figure()
    plt.plot(toxicity_loss, label="Tocixity Loss")
    plt.plot(toxicity_val_loss, label="Tocixity val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Toxicity Training /Validation Loss")
    # Filename <datetime>_loss_graph.png
    plt.savefig(
        f'{BASE_PATH}/logs/losses/{title}_{datetime.now().strftime("%Y%m%d%H%M%S")}_loss_graph.png'
    )

def show_emotion_loss_graph(emotion_loss,emotion_val_loss, title: str):
    """
    Shows the graph of the losses, save in photo
    """
    plt.figure()
    plt.plot(emotion_loss, label="Emotion Loss")
    plt.plot(emotion_val_loss, label="Emotion val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Emotion Training /Validation Loss")
    # Filename <datetime>_loss_graph.png
    plt.savefig(
        f'{BASE_PATH}/logs/losses/{title}_{datetime.now().strftime("%Y%m%d%H%M%S")}_loss_graph.png'
    )

def show_accuracy_graph(accuracy, val_accuracy, title: str):
    """
    Shows the graph of the accuracy, save in photo
    """
    plt.figure()
    plt.plot(accuracy, label="Accuracy")
    plt.plot(val_accuracy, label="val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy model")
    # Filename <datetime>_loss_graph.png
    plt.savefig(
        f'{BASE_PATH}/logs/accuracy/{title}_{datetime.now().strftime("%Y%m%d%H%M%S")}_accuracy_graph.png'
    )

def show_toxicity_accuracy_graph(toxicity_accuracy, toxicity_val_accuracy, title: str):
    """
    Shows the graph of the accuracy, save in photo
    """
    plt.figure()
    plt.plot(toxicity_accuracy, label="Toxicity Accuracy")
    plt.plot(toxicity_val_accuracy, label="Toxicity val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Toxicity")
    # Filename <datetime>_loss_graph.png
    plt.savefig(
        f'{BASE_PATH}/logs/accuracy/{title}_{datetime.now().strftime("%Y%m%d%H%M%S")}_accuracy_graph.png'
    )

def show_emotion_accuracy_graph(emotion_accuracy, emotion_val_accuracy, title: str):
    """
    Shows the graph of the accuracy, save in photo
    """
    plt.figure()
    plt.plot(emotion_accuracy, label="Emotion Accuracy")
    plt.plot(emotion_val_accuracy, label="Emotion val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Emotion")
    # Filename <datetime>_loss_graph.png
    plt.savefig(
        f'{BASE_PATH}/logs/accuracy/{title}_{datetime.now().strftime("%Y%m%d%H%M%S")}_accuracy_graph.png'
    )

def calculate_accuracy(y_true, y_pred):
    """
    Changed the name from `accuracy_score`
    Reference:  https://stackoverflow.com/questions/43962599/accuracy-score-in-pytorch-lstm
    """
    y_pred = np.concatenate(tuple(y_pred))
    y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(
        y_pred.shape
    )
    return (y_true == y_pred).sum() / float(len(y_true))

def save_to_json(scores, filename):
    """
    Saves the scores to a json file
    """
    import json

    with open(filename, "w") as f:
        json.dump(str(scores), f)


def log_hyperparameter(**kwargs):
    """
    Log the hyperparameters to a json file
    """

    for key, value in kwargs.items():
        print(f"{key}: {value}")

    from datetime import datetime
    
    try:
        date = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        with open(f"{BASE_PATH}/logs/{date}_hyperparameters.json", "w") as f:
            f.write(str(kwargs))
    except FileNotFoundError :
        raise FileNotFoundError("File not found, {BASE_PATH}/logs/{date}_hyperparameters.json")

# for running the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_length = 65  # Adjust as needed

def encode_texts(text):
    return tokenizer.encode(text,add_special_tokens=True, 
                                                max_length=65, 
                                                padding='max_length', 
                                                return_attention_mask=False,
                                                # truncation=True,
                                                return_tensors='tf')
class Decoder:
    def __init__(self, data_path_name):
        import pandas as pd
        df = pd.read_pickle(f'{BASE_PATH}\\dataset\\{data_path_name}')
        self.encoder_emotion = OneHotEncoder(sparse_output=False)
        self.encoder_toxicity = OneHotEncoder(sparse_output=False)

        self.encoder_emotion.fit_transform(df[['emotion']])
        self.encoder_toxicity.fit_transform(df[['toxicity']])
    
    # Decoding one-hot encoded labels
    
    def decode_toxicity(self,pred):
        return self.encoder_toxicity.inverse_transform(pred)
    
    def decode_emotion(self,pred):
        return self.encoder_emotion.inverse_transform(pred)
