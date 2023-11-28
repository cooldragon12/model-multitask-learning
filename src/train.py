"""
Train the model

function includes the finetuning and training of the model.
"""

from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
import torch

from build.pytorch.model import MultiTaskModel
from build.pytorch.preprocessing import ValorantChatDataset
from torch.utils.data import DataLoader
from utils.config import BASE_PATH, learning_rate, batch_size, num_epochs, num_layers, dropout, pre_trained_language_model, weight_decay
from utils.utils import train_fn, evaluate_fn, select_device, GridSearch





def run_training_with_grid_search():
    """
    Run the model with grid search

    This function runs the model with grid search. The grid search is implemented in the `GridSearch` class.
    """
    try:
        df = pd.read_pickle(f'{BASE_PATH}\\dataset\\preprocessed_df.pkl')
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        # Run the grid search
        gs = GridSearch(
                model=MultiTaskModel,
                train_set=train_data, 
                test_set=test_data, 
                batch_size=batch_size, 
                learning_rate=learning_rate, 
                num_epochs=num_epochs, 
                num_layers=num_layers, 
                dropout=dropout, 
                pre_trained_language_model=pre_trained_language_model,
                weight_decay=weight_decay
            )
        # Run the grid search
        gs.fit()

    except Exception as e:
        raise e

def run_training_with_hyperparameter_tuning(epochs: int, batch_size: int, learning_rate: float, num_layers: int, dropout: float, weight_decay: float = 0):
    try: 
        device = select_device('dml')
        TRANSFORMER = 'bert-base-uncased'
        # Tokenizer
        TOKENIZER = BertTokenizer.from_pretrained(TRANSFORMER)
        # Pytorch model
        model = MultiTaskModel(TRANSFORMER, dropout=dropout, batch_size=batch_size, num_layers=num_layers)
        # Loss function
        criterion = nn.CrossEntropyLoss()
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Load the preprocessed dataframe
        df = pd.read_pickle(f'{BASE_PATH}\\dataset\\preprocessed_df.pkl')
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

        # Create train and test datasets
        train_dataset = ValorantChatDataset(train_data['chat'].values, train_data['emotion'].values, train_data['toxicity'].values, TOKENIZER)
        test_dataset = ValorantChatDataset(test_data['chat'].values, test_data['emotion'].values, test_data['toxicity'].values, TOKENIZER)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        # Display the model hyperparameters
        print(f'Batch Size: {batch_size}, Learning Rate: {learning_rate}, Num Layers: {num_layers}, Dropout: {dropout}')
        for epoch in range(1, epochs + 1):
            # Train the model for one epoch
            loss, t_loss, e_loss= train_fn(model, criterion, optimizer, train_dataloader,device, epoch, epochs)
            # Evaluate the model on the test set
            at,ae,pt,pe,rt,re,f1t,f1e,cmt,cme= evaluate_fn(model, test_dataloader, device)

            print(f'Toxicity Accuracy: {at}, Emotion Accuracy: {ae}')
        # Save the model
        torch.save(model.state_dict(), f'{BASE_PATH}\\checkpoints\\pytorch\\mtm_with_{TRANSFORMER}_{datetime.now().date().__str__()}_{datetime.now().timestamp().__str__().split(".")[0]}_model.pt')
    except Exception as e:
        print(e)
