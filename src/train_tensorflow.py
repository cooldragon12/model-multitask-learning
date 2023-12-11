from re import T
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, Dense
from keras.regularizers import l2, l1_l2
from utils.config import BASE_PATH, TOXICITY_LABELS, EMOTIONS_LABELS
from datetime import datetime
from utils.utils import (
    log_hyperparameter,
    save_to_json
    )
def train_with_hyperparameter(epochs, batch_size, learning_rate, lstm_layers, dropout, l2_emotion = 0.032, l2_toxicity = 0.028,l2_lstm=0.01, learning_decay = 0.1, weight_epoch = 5):
    def lr_schedule(epoch, lr):
        if epoch % weight_epoch == 0 and epoch != 0:
            return lr * learning_decay
        else:
            return lr

    # Define the model checkpoint callback
    model_checkpoint = ModelCheckpoint(
        filepath=f'{BASE_PATH}/checkpoints/tensorflow/checkpoint_mtm_with_bert_{epochs}_{batch_size}_{learning_rate}_{lstm_layers}_{dropout}_l2_{l2_lstm}.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=5
    )
    # Log the hyperparameters
    log_hyperparameter(epochs, batch_size, learning_rate, lstm_layers, dropout, l2_emotion, l2_toxicity, l2_lstm)
    
    layer_1, layer_2 = lstm_layers, lstm_layers//2

    print("Initiating the dataset")
    df_combined = pd.read_pickle(f'{BASE_PATH}\\dataset\\preprocessed_df_combined.pkl')

    print("Initiating the tokenizer")
    # Preprocess the data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 65  # Adjust as needed

    def encode_texts(texts):
        return np.array(tokenizer.batch_encode_plus(texts, 
                                                    add_special_tokens=True, 
                                                    max_length=max_length, 
                                                    padding=True, 
                                                    return_attention_mask=False,
                                                    truncation=True,
                                                    return_tensors='tf')["input_ids"])
    
    X = encode_texts(df_combined['chat'].values)

    # One-hot encode labels
    encoder_emotion = OneHotEncoder(sparse_output=False)
    encoder_toxicity = OneHotEncoder(sparse_output=False)

    y_emotion_combined = encoder_emotion.fit_transform(df_combined[['emotion']])
    y_toxicity_combined = encoder_toxicity.fit_transform(df_combined[['toxicity']])


    X_train, X_test, y_train_emotion, y_test_emotion, y_train_toxicity, y_test_toxicity = train_test_split(X, y_emotion_combined,y_toxicity_combined, test_size=0.2) 
    # initilize optimizer
    print("Initiating the optimizer")
    initial_learning_rate = learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    # Define the learning rate scheduler
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    # Load BERT\
    print("Initiating the BERT")
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    print("Creating the model")
    # Model definition inside the loop
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    bert_output = bert(input_ids)[0] # type: ignore

    bi_lstm = Bidirectional(LSTM(layer_1, return_sequences=True, dropout=dropout, kernel_regularizer=l2(l2_lstm)))(bert_output)
    bi_lstm = Bidirectional(LSTM(layer_2, dropout=dropout*0.5))(bi_lstm)

    output_emotion = Dense(y_emotion_combined.shape[1], activation='softmax', name='emotion_output', kernel_regularizer=l1_l2(l2_emotion,l2_emotion))(bi_lstm)
    output_toxicity = Dense(y_toxicity_combined.shape[1], activation='softmax', name='toxicity_output',kernel_regularizer=l1_l2(l2_toxicity,l2_toxicity))(bi_lstm)

                        
    model = Model(inputs=input_ids, outputs=[output_emotion, output_toxicity])
                    # # Compile
                    # model = create_multitask_model_with_bert(y_toxicity, y_emotion, TFBertModel, max_length, lstm_dropout=0.2, layers=lstm_layers)
    model.compile(
        optimizer=optimizer, 
        loss={'emotion_output': 'categorical_crossentropy', 'toxicity_output': 'categorical_crossentropy'}, 
        metrics={
            'emotion_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='em_auc', multi_label=True)], 
            'toxicity_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='to_auc', multi_label=True)], 
            }
        
    )
    model.summary()
    print("Training the model")
                    # Train
    history = model.fit(
        X_train, {'emotion_output': y_train_emotion, 'toxicity_output': y_train_toxicity}, 
        validation_data=(X_test, {'emotion_output': y_test_emotion, 'toxicity_output': y_test_toxicity}), 
        epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler, model_checkpoint, early_stopping])
    print("Saving the model......")
    model.save(f'bi_lstm_bert_model_{epochs}_{batch_size}_adam_{dropout}_{layer_1}_{layer_2}_l1l2_{l2_emotion}_{l2_toxicity}.h5')
    # Show the results
    try:

        print("Saving the results")
        # Format of file 'dat'
        save_to_json(history.history, f'{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}_mtm_with_bert_{epochs}_{batch_size}_adam_{layer_1}_{layer_2}_with_regularization_history.json')
    except Exception as e:
        print(e)
        print("Error while saving the results")
    print("Done")
# Load your CSV file
def grid_search_train(epochs_options, batch_size_options, dropouts_options, lstm_layers_options, learning_rate_options, l2_emotion_options, l2_toxicity_options, l2_lstm_options, learning_decay_options, weight_epoch_options):
    # Grid search
    for epochs in epochs_options:
        for batch_size in batch_size_options:
            for dropout in dropouts_options:
                for lstm_layers in lstm_layers_options:
                    for learning_rate in learning_rate_options:
                        for l2_emotion in l2_emotion_options:
                            for l2_toxicity in l2_toxicity_options:
                                for l2_lstm in l2_lstm_options:
                                    for learning_decay in learning_decay_options:
                                        for weight_epoch in weight_epoch_options:
                                            print("Initiating the dataset")
                                            train_with_hyperparameter(
                                                epochs= epochs,
                                                batch_size=batch_size,
                                                learning_rate=learning_rate,
                                                lstm_layers=lstm_layers,
                                                dropout=dropout,
                                                weight_epoch=weight_epoch,
                                                l2_emotion=l2_emotion,
                                                l2_toxicity=l2_toxicity,
                                                learning_decay=learning_decay,
                                                l2_lstm=l2_lstm
                                            )
                                            print("Done")

                    # Optionally, save the model or log the results after each iteration
