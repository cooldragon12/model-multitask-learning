import tensorflow as tf
from transformers import TFRobertaModel, RobertaConfig
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, Dense


def create_multitask_model_with_bert(y_toxicity, y_emotion, bert_backbone, max_token_length=100, lstm_dropout=0.2, layers=(64, 32)):
    """
    Creates a model using BERT as backbone and Bi-LSTM layers.

    Args:
        y_toxicity (np.ndarray): One-hot encoded toxicity labels.
        y_emotion (np.ndarray): One-hot encoded emotion labels.
        max_token_length (int): The maximum length of input tokens.
        bert_dropout (float, optional): Dropout rate for BERT layer. Defaults to 0.2.
        lstm_dropout (float, optional): Dropout rate for LSTM layers. Defaults to 0.2.
        layers (tuple, optional): Number of units in each LSTM layer. Defaults to (64, 32).

    Returns:
        tf.keras.Model: The created model.

    """
    # Create the input layer
    input_ids = Input(shape=(max_token_length,), dtype=tf.int32, name='input_ids')
    
    # Configure BERT with the specified dropout rate
    
    # Load the BERT model with the base uncased configuration
    bert = bert_backbone.from_pretrained('bert-base-uncased')
    
    # Get the BERT pooled output
    pooled_output = bert(input_ids)[1] # type: ignore

    # Apply Bi-LSTM layer with return_sequences=True to get sequence outputs
    bi_lstm = Bidirectional(LSTM(layers[0], return_sequences=True, dropout=lstm_dropout))(pooled_output)
    
    # Apply another Bi-LSTM layer without return_sequences to get final output
    bi_lstm = Bidirectional(LSTM(layers[1]))(bi_lstm)
    
    # Output layer for emotion classification
    output_emotion = Dense(y_emotion.shape[1], activation='softmax', name='emotion_output')(bi_lstm)
    
    # Output layer for toxicity classification
    output_toxicity = Dense(y_toxicity.shape[1], activation='softmax', name='toxicity_output')(bi_lstm)

    # Create the model with input and output layers
    model = Model(inputs=input_ids, outputs=[output_emotion, output_toxicity])

    return model

def create_multitask_model_with_roberta(y_toxicity, y_emotion, max_token_length, bert_dropout=0.2, lstm_dropout=0.2, layers=(64, 32)):
    """
    Creates a model using RoBERTa as backbone and Bi-LSTM layers.

    Returns:
        tf.keras.Model: The created model.

    """
    # Create the input layer
    input_ids = Input(shape=(max_token_length,), dtype=tf.int32, name='input_ids')
    
    # Configure RoBERTa with the specified dropout rate
    config = RobertaConfig(dropout=bert_dropout)
    
    # Load the RoBERTa model with the base uncased configuration
    roberta = TFRobertaModel.from_pretrained('roberta-base', config=config)
    
    # Get the RoBERTa pooled output
    pooled_output = roberta(input_ids)[1] # type: ignore

    # Apply Bi-LSTM layer with return_sequences=True to get sequence outputs
    bi_lstm = Bidirectional(LSTM(layers[0], return_sequences=True, dropout=lstm_dropout))(pooled_output)
    
    # Apply another Bi-LSTM layer without return_sequences to get final output
    bi_lstm = Bidirectional(LSTM(layers[1]))(bi_lstm)
    
    # Output layer for emotion classification
    output_emotion = Dense(y_emotion.shape[1], activation='softmax', name='emotion_output')(bi_lstm)
    
    # Output layer for toxicity classification
    output_toxicity = Dense(y_toxicity.shape[1], activation='softmax', name='toxicity_output')(bi_lstm)

    # Create the model with input and output layers
    model = Model(inputs=input_ids, outputs=[output_emotion, output_toxicity])

    return model