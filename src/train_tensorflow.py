import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, Dense
from keras.regularizers import l2
from utils.config import BASE_PATH

from utils.utils import (
    show_accuracy_graph, 
    show_emotion_accuracy_graph, 
    show_emotion_loss_graph, 
    show_loss_graph, 
    show_toxicity_accuracy_graph, 
    show_toxicity_loss_graph,
    save_to_json
    )
# Load your CSV file
def grid_search_train():
    df = pd.read_pickle(f'{BASE_PATH}\\dataset\\preprocessed_df_original.pkl')

    # Preprocess the data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128  # Adjust as needed

    def encode_texts(texts):
        return np.array(tokenizer.batch_encode_plus(texts, 
                                                    add_special_tokens=True, 
                                                    max_length=max_length, 
                                                    padding=True, 
                                                    return_attention_mask=False,
                                                    truncation=True,
                                                    return_tensors='tf')["input_ids"])

    X = encode_texts(df['chat'].values)

    # One-hot encode labels
    encoder_emotion = OneHotEncoder(sparse_output=False)
    encoder_toxicity = OneHotEncoder(sparse_output=False)

    y_emotion = encoder_emotion.fit_transform(df[['emotion']])
    y_toxicity = encoder_toxicity.fit_transform(df[['toxicity']])

    # Split the dataset
    X_train, X_test, y_train_emotion, y_test_emotion, y_train_toxicity, y_test_toxicity = train_test_split(X, y_emotion, y_toxicity, test_size=0.2, random_state=42)

    # Load BERT
    bert = TFBertModel.from_pretrained('bert-base-uncased')

    # Hyperparameter gridchatgpt
    epochs_options = [10,20, 50]
    batch_size_options = [32, 64]
    optimizer = 'adam'
    lstm_layers_options = [(64, 32)]
    dropouts = [0.2, 0.3, 0.4, 0.5]
    # Grid search
    for epochs in epochs_options:
        for batch_size in batch_size_options:
            for dropout in dropouts:
                for lstm_layers in lstm_layers_options:
                    # Model definition inside the loop
                    input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
                    bert_output = bert(input_ids)[0] # type: ignore

                    bi_lstm = Bidirectional(LSTM(lstm_layers[0], return_sequences=True, dropout=dropout))(bert_output)
                    bi_lstm = Bidirectional(LSTM(lstm_layers[1]))(bi_lstm)

                    output_emotion = Dense(y_emotion.shape[1], activation='softmax', name='emotion_output')(bi_lstm)
                    output_toxicity = Dense(y_toxicity.shape[1], activation='softmax', name='toxicity_output')(bi_lstm)

                        
                    model = Model(inputs=input_ids, outputs=[output_emotion, output_toxicity])
                    # # Compile
                    # model = create_multitask_model_with_bert(y_toxicity, y_emotion, TFBertModel, max_length, lstm_dropout=0.2, layers=lstm_layers)
                    model.compile(optimizer=optimizer, 
                                    loss={'emotion_output': 'categorical_crossentropy', 'toxicity_output': 'categorical_crossentropy'}, 
                                    metrics={'emotion_output': 'accuracy', 'toxicity_output': 'accuracy'})

                    # Train
                    history = model.fit(X_train, {'emotion_output': y_train_emotion, 'toxicity_output': y_train_toxicity}, 
                                            validation_data=(X_test, {'emotion_output': y_test_emotion, 'toxicity_output': y_test_toxicity}), 
                                            epochs=epochs, batch_size=batch_size)
                    # Show the results
                    print("Saving the model.....")
                    model.save(f'bi_lstm_bert_model_{epochs}_{batch_size}_{optimizer}_{dropout}_{lstm_layers[0]}_{lstm_layers[1]}.h5')
                    print("Saving the results....")
                    save_to_json(history.history, f'mtm_with_bert_{epochs}_{batch_size}_{optimizer}_{lstm_layers[0]}_{lstm_layers[1]}_history.json')
                    show_loss_graph(history.history['loss'], history.history['val_loss'], f'mtm_with_bert_{epochs}_{batch_size}_{optimizer}_{lstm_layers[0]}_{lstm_layers[1]}_val_loss_output')
                    show_emotion_loss_graph(history.history['emotion_output_loss'], history.history['val_emotion_output_loss'], f'mtm_with_bert_{epochs}_{batch_size}_{optimizer}_{lstm_layers[0]}_{lstm_layers[1]}_val_emotion_output_loss')
                    show_toxicity_loss_graph(history.history['toxicity_output_loss'], history.history['val_toxicity_output_loss'], f'mtm_with_bert_{epochs}_{batch_size}_{optimizer}_{lstm_layers[0]}_{lstm_layers[1]}_val_toxicity_output_loss')
                    show_toxicity_accuracy_graph(history.history['toxicity_output_accuracy'], history.history['val_toxicity_output_accuracy'], f'mtm_with_bert_{epochs}_{batch_size}_{optimizer}_{lstm_layers[0]}_{lstm_layers[1]}_val_toxicity_output_accuracy')
                    show_emotion_accuracy_graph(history.history['emotion_output_accuracy'], history.history['val_emotion_output_accuracy'], f'mtm_with_bert_{epochs}_{batch_size}_{optimizer}_{lstm_layers[0]}_{lstm_layers[1]}_val_emotion_output_accuracy')
                    show_accuracy_graph(history.history['val_emotion_output_accuracy'], history.history['val_toxicity_output_accuracy'], f'mtm_with_bert_{epochs}_{batch_size}_{optimizer}_{lstm_layers[0]}_{lstm_layers[1]}_val_output_accuracy')
                    # Save the results
                    # Optionally, save the model or log the results after each iteration

def train_with_hyperparameter(epochs, batch_size, learning_rate, lstm_layers, dropout):
    print(
        f"""
        Hyperparameters:
        epochs: {epochs}
        batch_size: {batch_size}
        learning_rate: {learning_rate}
        lstm_layers: {lstm_layers}
        dropout: {dropout}

        applied l2 regularization:
        Task Emotion: 0.015
        Task Toxicity: 0.03
        """)
    layer_1, layer_2 = lstm_layers, lstm_layers//2
    print("Initiating the dataset")
    df = pd.read_pickle(f'{BASE_PATH}\\dataset\\preprocessed_df_original.pkl')

    print("Initiating the tokenizer")
    # Preprocess the data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128  # Adjust as needed

    def encode_texts(texts):
        return np.array(tokenizer.batch_encode_plus(texts, 
                                                    add_special_tokens=True, 
                                                    max_length=max_length, 
                                                    padding=True, 
                                                    return_attention_mask=False,
                                                    truncation=True,
                                                    return_tensors='tf')["input_ids"])

    X = encode_texts(df['chat'].values)

    # One-hot encode labels
    encoder_emotion = OneHotEncoder(sparse_output=False)
    encoder_toxicity = OneHotEncoder(sparse_output=False)

    y_emotion = encoder_emotion.fit_transform(df[['emotion']])
    y_toxicity = encoder_toxicity.fit_transform(df[['toxicity']])
    print("Splitting the dataset")
    # Split the dataset
    X_train, X_test, y_train_emotion, y_test_emotion, y_train_toxicity, y_test_toxicity = train_test_split(X, y_emotion, y_toxicity, test_size=0.2, random_state=42)

    # initilize optimizer
    print("Initiating the optimizer")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Load BERT\
    print("Initiating the BERT")
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    print("Creating the model")
    # Model definition inside the loop
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    bert_output = bert(input_ids)[0] # type: ignore

    bi_lstm = Bidirectional(LSTM(layer_1, return_sequences=True, dropout=dropout))(bert_output)
    bi_lstm = Bidirectional(LSTM(layer_2))(bi_lstm)

    output_emotion = Dense(y_emotion.shape[1], activation='softmax', name='emotion_output', kernel_regularizer=l2(0.015))(bi_lstm)
    output_toxicity = Dense(y_toxicity.shape[1], activation='softmax', name='toxicity_output',kernel_regularizer=l2(0.03))(bi_lstm)

                        
    model = Model(inputs=input_ids, outputs=[output_emotion, output_toxicity])
                    # # Compile
                    # model = create_multitask_model_with_bert(y_toxicity, y_emotion, TFBertModel, max_length, lstm_dropout=0.2, layers=lstm_layers)
    model.compile(optimizer=optimizer, 
                                    loss={'emotion_output': 'categorical_crossentropy', 'toxicity_output': 'categorical_crossentropy'}, 
                                    metrics={'emotion_output': 'accuracy', 'toxicity_output': 'accuracy'})
    print("Training the model")
                    # Train
    history = model.fit(X_train, {'emotion_output': y_train_emotion, 'toxicity_output': y_train_toxicity}, 
                                            validation_data=(X_test, {'emotion_output': y_test_emotion, 'toxicity_output': y_test_toxicity}), 
                                            epochs=epochs, batch_size=batch_size)
    print("Saving the model......")
    model.save(f'bi_lstm_bert_model_{epochs}_{batch_size}_adam_{dropout}_{layer_1}_{layer_2}.h5')
    # Show the results
    try:
        print("Saving the results")
        save_to_json(history.history, f'mtm_with_bert_{epochs}_{batch_size}_adam_{layer_1}_{layer_2}_with_regularization_history.json')
        show_loss_graph(history.history['loss'], history.history['val_loss'], f'mtm_with_bert_{epochs}_{batch_size}_adam_{layer_1}_{layer_2}_val_loss_output')
        show_emotion_loss_graph(history.history['emotion_output_loss'], history.history['val_emotion_output_loss'], f'mtm_with_bert_{epochs}_{batch_size}_adam_{layer_1}_{layer_2}_val_emotion_output_loss')
        show_toxicity_loss_graph(history.history['toxicity_output_loss'], history.history['val_toxicity_output_loss'], f'mtm_with_bert_{epochs}_{batch_size}_adam_{layer_1}_{layer_2}_val_toxicity_output_loss')
        show_toxicity_accuracy_graph(history.history['toxicity_output_accuracy'], history.history['val_toxicity_output_accuracy'], f'mtm_with_bert_{epochs}_{batch_size}_adam_{layer_1}_{layer_2}_val_toxicity_output_accuracy')
        show_emotion_accuracy_graph(history.history['emotion_output_accuracy'], history.history['val_emotion_output_accuracy'], f'mtm_with_bert_{epochs}_{batch_size}_adam_{layer_1}_{layer_2}_val_emotion_output_accuracy')
        show_accuracy_graph(history.history['val_emotion_output_accuracy'], history.history['val_toxicity_output_accuracy'], f'mtm_with_bert_{epochs}_{batch_size}_adam_{layer_1}_{layer_2}_val_output_accuracy')
                        # Save the results
                        # Optionally, save the model or log the results after each iteration
    except Exception as e:
        print(e)
        print("Error while saving the results")
    print("Done")