from datetime import datetime
from sympy import hyper
import tqdm
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
from .config import BASE_PATH




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


def log_hyperparameter(epoch, batch_size, learning_rate, lstm_layers, dropout, l2_emotion, l2_toxicity, l2_lstm):
    """
    Log the hyperparameters to a json file
    """
    print(
        f"""
        Hyperparameters:
        epochs: {epoch}
        batch_size: {batch_size}
        learning_rate: {learning_rate}
        lstm_layers: {lstm_layers}
        dropout: {dropout}

        applied l2 regularization:
        Task Emotion: {l2_emotion}
        Task Toxicity: {l2_toxicity}
        LSTM: {l2_lstm}
        """)
    from datetime import datetime
    import json
    hyperparameters ={"epochs": epoch,"batch_size": batch_size,"learning_rate": learning_rate,"lstm_layers": lstm_layers,"dropout": dropout,"l2_emotion": l2_emotion,"l2_toxicity": l2_toxicity,"l2_lstm": l2_lstm,"date": datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}
    
    try:
        date = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        with open(f"{BASE_PATH}/logs/{date}_hyperparameters.json", "w") as f:
            f.write(str(hyperparameters))
    except FileNotFoundError :
        raise FileNotFoundError("File not found, {BASE_PATH}/logs/{date}_hyperparameters.json")


