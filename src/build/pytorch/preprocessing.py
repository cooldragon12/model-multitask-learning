from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.config import EMOTIONS_LABELS, TOXICITY_LABELS
MAX_LEN = 256 # Define the maximum length of tokenized texts

class ValorantChatDataset(Dataset):
    """
    Structuring the custom dataset for the Valorant Chat Dataset
    
    with multiple target variables:
    X variable
    - Chat

    Target Variables
    - Toxicity Labels
    - Emotion Labels


    Reference: https://niteshkumardew11.medium.com/fine-tuning-bert-base-using-pytorch-for-sentiment-analysis-c44a3ce79091
    """
    
    def __init__(self, chats, e_labels, t_labels,tokenizer,  max_len=256):
        self.tokenizer = tokenizer
        self.chats = chats # Text chat
        self.e_labels_unencoded = e_labels # Emotion labels
        self.t_labels_unencoded = t_labels # Toxicity labels
        # NOTE: This is depends on the max length of the chat text, if possible close it to the max length of the chat text
        self.max_len = max_len # Max length of the tokenized text
        # One hot encode the labels
        # self.one_hot_encoder_emotion = OneHotEncoder(sparse=False)
        # self.one_hot_encoder_toxicity = OneHotEncoder(sparse=False)
        # # Fit the one hot encoder
        # self.one_hot_encoder_emotion.fit(self.EMOTIONS_LABELS.reshape(-1,1))
        # self.one_hot_encoder_toxicity.fit(self.TOXICITY_LABELS.reshape(-1,1))
    # def _category_to_one_hot(self):
    #     """
    #     Convert the labels to one hot encoding
    #     """
    #     emotion = self.one_hot_encoder_emotion.categories_
    #     toxicity = self.one_hot_encoder_toxicity.categories_
    #     return emotion, toxicity
    def __len__(self):
        return len(self.chats)
    
    def __getitem__(self, index):
        chat = self.chats[index]
        encoded = self.tokenizer.encode_plus(
            chat,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        emotion_label = (EMOTIONS_LABELS.index(self.e_labels_unencoded[index]))
        toxicity_label = (TOXICITY_LABELS.index(self.t_labels_unencoded[index]))

        # e_labels = self.one_hot_encoder_emotion.transform(np.array(self.e_labels_unencoded[index]).reshape(-1,1))
        # t_labels = self.one_hot_encoder_toxicity.transform(np.array(self.t_labels_unencoded[index]).reshape(-1,1))
        e_labels = F.one_hot(torch.tensor(emotion_label), num_classes=3).float()
        t_labels = F.one_hot(torch.tensor(toxicity_label), num_classes=17).float()
        # Returns the encoded chat, input ids, attention mask, emotion labels and toxicity labels
        return {
            'chat': chat,
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'emotion_labels': e_labels,
            'toxicity_labels': t_labels,
            'raw_emotion_labels': self.e_labels_unencoded[index],
            'raw_toxicity_labels': self.t_labels_unencoded[index]
        }