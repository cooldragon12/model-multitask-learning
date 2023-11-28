from torch.autograd import Variable
import torch
from transformers import BertModel, RobertaModel
import torch.nn as nn
from torch.nn import functional as F


class MultiTaskModel(nn.Module):
    """
    MultiTaskModel is a PyTorch module that implements a multi-task learning model using BERT as the backbone.
    It consists of two LSTM classifiers for toxicity and emotion classification, respectively.
    """

    def __init__(self, backbone='bert-base-uncased', dropout=0.3, batch_size=16, num_layers=1):
        super(MultiTaskModel, self).__init__()
        self.num_layers = num_layers
        self.input_size = 768 # Default for bert-base-uncased
        self.batch_size = batch_size
        self.hidden_size_lstm = 256

        # Backbone of the model: BERT
        if backbone == 'bert-base-uncased':
            self.bert_embedded = BertModel.from_pretrained(
                "bert-base-uncased", output_hidden_states=False
            )
        elif backbone == 'roberta-base':
            self.bert_embedded = RobertaModel.from_pretrained(
                "roberta-base", output_hidden_states=False
            )
        else:
            raise ValueError('Backbone not supported')
        
        self.hidden_size = self.bert_embedded.config.hidden_size  # type: ignore

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.toxicity_classifier = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size_lstm,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        self.emotion_classifier = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size_lstm,
            num_layers=self.num_layers,
            bidirectional=True,
        )

        self.toxicity_hidden = nn.Linear(self.hidden_size_lstm * 2, 17)
        self.emotion_hidden = nn.Linear(self.hidden_size_lstm * 2, 3)

        # self.toxicity_hidden_2 = nn.Linear(self.hidden_size_lstm, 17)
        # self.emotion_hidden_2 = nn.Linear(self.hidden_size_lstm, 3)

        self.toxicity_softmax = nn.Softmax(dim=1)
        self.emotion_softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU()

        # self.hidden = self.init_hidden()

    # def init_hidden(self):
    #     # first is the hidden h
    #     # second is the cell c
    #     return (
    #         Variable(
    #             torch.zeros(self.num_layers, self.batch_size, self.hidden_size_lstm)
    #         ),
    #         Variable(
    #             torch.zeros(self.num_layers, self.batch_size, self.hidden_size_lstm)
    #         ),
    #     )

    def bertPooled(self, inputs, attention_mask):
        """
        Apply BERT embedding and pooling to the input sequence.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Pooled output tensor of shape (batch_size, hidden_size).
        """
        # Run backbone layer
        outputs = self.bert_embedded(inputs, attention_mask=attention_mask, return_dict=False)  # type: ignore
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return pooled_output

    def classifier(self, output):
        """
        Apply LSTM classifiers to the input sequence.

        Args:
            output (torch.Tensor): Input tensor of shape (batch_size, hidden_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing toxicity logits and emotion logits.
        """
        # Use only the last hidden state for classification
        toxicity_logits, (h_out_t, _) = self.toxicity_classifier(output)
        emotion_logits, (h_out_e, _) = self.emotion_classifier(output)
        return toxicity_logits, emotion_logits, h_out_t, h_out_e

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the multi-task learning model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing toxicity logits,
            emotion logits, toxicity probabilities, and emotion probabilities.
        """
        # Backbone Layer
        pooled_output = self.bertPooled(input_ids, attention_mask)
        # Use the features from the pooled output from BERT
        toxicity_logits, emotion_logits, h_out_t, h_out_e = self.classifier(
            pooled_output
        )

        # hidden_toxicity = torch.cat((toxicity_logits[:,-1, :256],toxicity_logits[:,0, 256:]),dim=-1)
        # hidden_emotion = torch.cat((emotion_logits[:,-1, :256],emotion_logits[:,0, 256:]),dim=-1)
        # h_out_t = h_out_t.view(-1, self.hidden_size_lstm)
        # h_out_e = h_out_e.view(-1, self.hidden_size_lstm)
        # Use only the last hidden state for classification
        toxicity_y = self.toxicity_hidden(toxicity_logits)
        emotion_y = self.emotion_hidden(emotion_logits)

        # Use only the last hidden state for classification
        # toxicity_y = self.toxicity_hidden_2(toxicity_y)
        # emotion_y = self.emotion_hidden_2(emotion_y)

        # Softmax along dimension 1
        toxicity_probs = self.toxicity_softmax(toxicity_y)
        emotion_probs = self.emotion_softmax(emotion_y)

        return toxicity_y, emotion_y, toxicity_probs, emotion_probs
