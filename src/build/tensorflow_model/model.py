import tensorflow as tf
from transformers import TFRobertaModel, RobertaConfig, BertModel, TFBertModel
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, Dense
from keras.regularizers import l1_l2

def create_multitask_model_with_bert_merged(bert_backbone:str,dropout:float,learning_rate=1e-4, max_length=65, l2=0.01, layers=40):
    """
    Creates a model using BERT as backbone and Bi-LSTM layers.

    Args:
        bert_backbone (str): The name of the BERT backbone to use.
        max_length (int): The maximum length of the input sequences.
        dropout (float): The dropout rate to apply to the BERT backbone.
        learning_rate (float): The learning rate for the optimizer.
        l2 (float): The L2 regularization parameter.
        layers (int): The number of units in the Bi-LSTM layers.

    Returns:
        tf.keras.Model: The created model.

    """
    print("Initiating the BERT")
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    print("Initiating the optimizer")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # type: ignore
    # Load BERT\
    print("Building the model")
    # Model definition inside the loop
    print("Inserting theInput component")
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    print("Inserting the BERT component")
    bert_output = bert(input_ids)[0] # type: ignore
    print("Inserting the Bi-LSTM components")
    bi_lstm_emotion = Bidirectional(LSTM(layers, dropout=dropout, kernel_regularizer=l1_l2(l2*0.15,l2)))(bert_output)
    bi_lstm_toxicity = Bidirectional(LSTM(layers, dropout=dropout, kernel_regularizer=l1_l2(l2*0.2,l2)))(bert_output)

    print("Densing the output layers")
    output_emotion = Dense(6, activation='softmax', name='emotion_output')(bi_lstm_emotion)
    output_toxicity = Dense(7, activation='softmax', name='toxicity_output')(bi_lstm_toxicity)

    print("Compiling the model")
    model = Model(inputs=input_ids, outputs=[output_emotion, output_toxicity])
                    # # Compile
                    # model = create_multitask_model_with_bert(y_toxicity, y_emotion, TFBertModel, max_length, lstm_dropout=0.2, layers=lstm_layers)
    model.compile(
        optimizer=optimizer, 
        loss={'emotion_output': 'categorical_crossentropy', 'toxicity_output': 'categorical_crossentropy'}, 
        metrics={
            'emotion_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='em_auc', multi_label=True), tf.keras.metrics.F1Score(name='f1_score')], 
            'toxicity_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='to_auc', multi_label=True), tf.keras.metrics.F1Score(name='f1_score')],
        }
    )
    print("Model created successfully")
    return model

def create_multitask_model_with_bert(learning_rate, max_token_length=100, lstm_dropout=0.2, layers=(64, 32)):
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
    print("Initiating the BERT")
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    print("Initiating the optimizer")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Create the input layer
    input_ids = Input(shape=(max_token_length,), dtype=tf.int32, name='input_ids')
    
    # Get the BERT pooled output
    pooled_output = bert(input_ids)[1] # type: ignore

    # Apply Bi-LSTM layer with return_sequences=True to get sequence outputs
    bi_lstm = Bidirectional(LSTM(layers[0], return_sequences=True, dropout=lstm_dropout))(pooled_output)
    
    # Apply another Bi-LSTM layer without return_sequences to get final output
    bi_lstm = Bidirectional(LSTM(layers[1]))(bi_lstm)
    
    # Output layer for emotion classification
    output_emotion = Dense(6, activation='softmax', name='emotion_output')(bi_lstm)
    
    # Output layer for toxicity classification
    output_toxicity = Dense(7, activation='softmax', name='toxicity_output')(bi_lstm)

    # Create the model with input and output layers
    model = Model(inputs=input_ids, outputs=[output_emotion, output_toxicity])

    model.compile(
        optimizer=optimizer, 
        loss={'emotion_output': 'categorical_crossentropy', 'toxicity_output': 'categorical_crossentropy'}, 
        metrics={
            'emotion_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='em_auc', multi_label=True), tf.keras.metrics.F1Score(name='f1_score')], 
            'toxicity_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='to_auc', multi_label=True), tf.keras.metrics.F1Score(name='f1_score')],
        }
    )
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