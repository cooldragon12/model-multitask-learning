import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Bidirectional, LSTM
from keras.regularizers import l1_l2
import tensorflow as tf
from tensoflow.keras.utils import plot_model
from transformers import TFBertModel

def generate_model_structure(learning_rate=0.0001, lstm_layers=40, dropout=0.75,l2_lstm=0.03):
    layer_1 = lstm_layers
    max_length = 65  # Adjust as needed
    # initilize optimizer
    print("Initiating the optimizer")
    initial_learning_rate = learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    # Define the learning rate scheduler
    
    # Load BERT\
    print("Initiating the BERT")
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    print("Creating the model")
    # Model definition inside the loop
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    bert_output = bert(input_ids)[0] # type: ignore

    bi_lstm_emotion = Bidirectional(LSTM(layer_1, dropout=dropout, kernel_regularizer=l1_l2(l2_lstm*0.15,l2_lstm)))(bert_output)
    bi_lstm_toxicity = Bidirectional(LSTM(layer_1, dropout=dropout, kernel_regularizer=l1_l2(l2_lstm*0.2,l2_lstm)))(bert_output)

    output_emotion = Dense(6, activation='softmax', name='emotion_output')(bi_lstm_emotion)
    output_toxicity = Dense(7, activation='softmax', name='toxicity_output')(bi_lstm_toxicity)
                        
    model = Model(inputs=input_ids, outputs=[output_emotion, output_toxicity])
                    
    model.compile(
        optimizer=optimizer, 
        loss={'emotion_output': 'categorical_crossentropy', 'toxicity_output': 'categorical_crossentropy'}, 
        metrics={
            'emotion_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='em_auc', multi_label=True)], 
            'toxicity_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='to_auc', multi_label=True)], 
            }
        
    )
    plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,   
    )
    return 
