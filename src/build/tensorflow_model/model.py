
from multiprocessing import pool
import tensorflow as tf
from transformers import TFBertModel
from keras.layers import Input, Dropout, LSTM, Dense, Concatenate, Bidirectional
from keras.models import Model
# class MultiTaskModel(tf.keras.Model):
#     def __init__(self, dropout=0.5, hidden_size=128, input_size=768, num_layers=1):
#         super(MultiTaskModel, self).__init__()
#         self.num_layers = num_layers
#         self.input_size = input_size

#         # Backbone of the model: BERT
#         self.bert_embedded = TFBertModel.from_pretrained('bert-base-uncased')
#         self.hidden_size = self.bert_embedded.config.hidden_size

#         # Dropout
#         self.dropout = Dropout(dropout)

#         # Classifiers
#         self.shared_classifier = Bidirectional(LSTM(
#             units=self.hidden_size,
#             dropout=dropout,
#             stateful=True
#         ))

#         self.toxicity_classifier = Dense(1, activation='softmax')
#         self.emotion_classifier = Dense(1,activation='softmax')

#     def call(self, inputs, attention_mask):
#         # Backbone Layer
#         _, pooled_output = self.bert_embedded(inputs, attention_mask=attention_mask)

#         # Use the features from the pooled output from BERT
#         pooled_output = self.dropout(pooled_output)

#         # Shared LSTM Classifier
#         _, forward_h, _ = self.shared_classifier(pooled_output)
#         _, backward_h, _ = self.shared_classifier(pooled_output, initial_state=[forward_h])

#         # Concatenate forward and backward hidden states
#         concatenated_h = Concatenate()([forward_h, backward_h])

#         # Apply classifiers
#         toxicity_y = self.toxicity_classifier(concatenated_h)
#         emotion_y = self.emotion_classifier(concatenated_h)

#         return toxicity_y, emotion_y

def getMultiTaskModel(backbone=None, dropout=0.5, hidden_size=128, input_size=768, num_layers=1):

    input_ids = Input(shape=(input_size,), dtype="int32",
                                        name="input_ids")
    attention_mask = Input(shape=(input_size,), dtype="int32",
                                    name="attention_mask")
    segment_ids = Input(shape=(input_size,), dtype="int32",
                                        name="segment_ids")

    # Backbone Layer
    # Input Layers
    # BERT Layer
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    output = bert_model([input_wids, attention_mask, segment_ids]) # type: ignore
    pooled_output = output[1] # type: ignore
    pooled_output = Dropout(dropout)(pooled_output)
    # Use of BI-LSTM Classifier
    shared_classifier = Bidirectional(LSTM(
        units=hidden_size,
        dropout=dropout,
        stateful=True,
    
    ), name='shared_classifier')

    # Shared LSTM Classifier Layer
    bi_lstm_output = shared_classifier(pooled_output)


    # Task-specific layers
    emotion_output = Dense(NUM_EMOTION_CLASSES, activation='sigmoid', name='emotion_output')(bi_lstm_output)

    toxicity_input = Concatenate(name="concatination_of_blstm_emotion")([bi_lstm_output, emotion_output])

    toxicity_output = Dense(NUM_TOXICITY_CLASSES, activation='sigmoid', name='toxicity_output')(toxicity_input)

    # Model
    model = Model(inputs=[input_ids, attention_mask, segment_ids], outputs=[emotion_output, toxicity_output])

    return model


model = getMultiTaskModel()
model.summary()