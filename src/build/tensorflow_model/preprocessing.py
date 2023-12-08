import pandas as pd
import numpy as np


from sklearn.preprocessing import OneHotEncoder
def encode_texts(texts, tokenizer, max_length):
    return np.array(tokenizer.batch_encode_plus(texts, 
                                                add_special_tokens=True, 
                                                max_length=max_length, 
                                                padding=True, 
                                                return_attention_mask=False,
                                                truncation=True,
                                                return_tensors='tf',
                                                )["input_ids"])


# One-hot encode labels
encoder_emotion = OneHotEncoder(sparse_output=False)

encoder_toxicity = OneHotEncoder(sparse_output=False)


