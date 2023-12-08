# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
# import torch
# import torch.optim as optim
# import torch.nn as nn
# import torch_directml as td
# from transformers import BertTokenizer, RobertaTokenizer
# from build.pytorch.model import MultiTaskModel
# from build.pytorch.preprocessing import ValorantChatDataset
# def save_checkpoint(state, filename):
#     """
#     Saves model state to a checkpoint
#     """
#     DEFAULT_PATH = f"{BASE_PATH}/checkpoints/pytorch/"
#     print("=> Saving checkpoint")
#     torch.save(state, f"{DEFAULT_PATH}/{filename}")
#     with open(f"{BASE_PATH}/logs/checkpoint.txt", "w") as f:
#         f.write(f"{DEFAULT_PATH}/{filename}")


# def load_checkpoint(checkpoint, model, optimizer):
#     """
#     Loads model state from a checkpoint
#     """
#     print("=> Loading checkpoint")
#     with open(f"{BASE_PATH}/logs/checkpoint.txt", "r") as f:
#         checkpoint = torch.load(f.read())
#     model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer"])


# def split_train_test_valid(*data, test_size=0.2, valid_size=0.5, random_state=42):
#     """
#     Split the data into train, test and valid sets
#     """
#     splitted_1 = train_test_split(*data, test_size=test_size, random_state=random_state)
#     final_split = train_test_split(
#         *splitted_1, test_size=valid_size, random_state=random_state
#     )
#     return final_split


# def train_fn(model, criterion, optimizer, data_loader, device, epoch, total_epoch):
#     """
#     Trains the model for one epoch on the training set, specifically for the ValorantChatDataset
#     """
#     progress_bar = tqdm.tqdm(
#         data_loader, desc="Epoch {:1d}/{:2d}".format(epoch, total_epoch)
#     )
#     losses = []
#     emotion_losses = []
#     toxicity_losses = []
#     model.train()
#     model.to(device)
#     for batch in progress_bar:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         toxicity_labels = batch["toxicity_labels"].to(device)
#         emotion_labels = batch["emotion_labels"].to(device)

#         # For debugging purposes
#         # raw_toxicity_labels = batch['raw_toxicity_labels']
#         # raw_emotion_labels = batch['raw_emotion_labels']
#         optimizer.zero_grad()

#         # Forward pass
#         toxicity_y, emotion_y, tox_probs, emo_probs = model(input_ids, attention_mask)

#         # Calculate Loss
#         toxicity_loss = criterion(tox_probs, toxicity_labels)
#         emotion_loss = criterion(emo_probs, emotion_labels)
#         loss = toxicity_loss + emotion_loss
#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         # Append the losses
#         toxicity_losses.append(toxicity_loss.item())
#         emotion_losses.append(emotion_loss.item())
#         losses.append(loss.item())

#     print("Loss:", losses)
#     # Return the average loss
#     return (
#         sum(losses) / len(losses),
#         sum(toxicity_losses) / len(toxicity_losses),
#         sum(emotion_losses) / len(emotion_losses),
#     )


# # def evaluate_fn(model, data_loader, device):
# #     """
# #     Evaluates the model on the test set

# #     Returns:
# #         accuracy_toxicity: Accuracy of the model on the toxicity task
# #         accuracy_emotion: Accuracy of the model on the emotion task
# #     """
# #     model.eval()
# #     all_toxicity_probs = []
# #     all_emotion_probs = []
# #     all_toxicity_labels = []
# #     all_emotion_labels = []

# #     progress_bar = tqdm.tqdm(data_loader, desc="Evaluating the model")

# #     with torch.no_grad():
# #         for batch in progress_bar:
# #             input_ids = batch["input_ids"].to(device)
# #             attention_mask = batch["attention_mask"].to(device)
# #             toxicity_labels = batch["toxicity_labels"].to(device)
# #             emotion_labels = batch["emotion_labels"].to(device)

# #             toxicity_logits, emotion_logits, toxicity_probs, emotion_probs = model(
# #                 input_ids, attention_mask
# #             )

# #             all_toxicity_probs.extend(toxicity_probs.cpu().numpy())
# #             all_emotion_probs.extend(emotion_probs.cpu().numpy())
# #             all_toxicity_labels.extend(toxicity_labels.cpu().numpy())
# #             all_emotion_labels.extend(emotion_labels.cpu().numpy())

# #     # Calculate evaluation scores (you can use your own evaluation metric)
# #     # For example, you can use accuracy, F1 score, etc.
# #     # Here, I'm using accuracy as an example.
# #     at = accuracy_toxicity(
# #         torch.tensor(all_toxicity_probs), torch.tensor(all_toxicity_labels)
# #     )
# #     ae = accuracy_emotion(
# #         torch.tensor(all_emotion_probs), torch.tensor(all_emotion_labels)
# #     )

# #     pt = precision_toxicity(
# #         torch.tensor(all_toxicity_probs), torch.tensor(all_toxicity_labels)
# #     )
# #     pe = precision_emotion(
# #         torch.tensor(all_emotion_probs), torch.tensor(all_emotion_labels)
# #     )

# #     rt = recall_toxicity(
# #         torch.tensor(all_toxicity_probs), torch.tensor(all_toxicity_labels)
# #     )
# #     re = recall_emotion(
# #         torch.tensor(all_emotion_probs), torch.tensor(all_emotion_labels)
# #     )

# #     f1t = f1_toxicity(
# #         torch.tensor(all_toxicity_probs), torch.tensor(all_toxicity_labels)
# #     )
# #     f1e = f1_emotion(torch.tensor(all_emotion_probs), torch.tensor(all_emotion_labels))

# #     cmt = confusion_matrix_toxicity(
# #         torch.tensor(all_toxicity_probs), torch.tensor(all_toxicity_labels)
# #     )
# #     cme = confusion_matrix_emotion(
# #         torch.tensor(all_emotion_probs), torch.tensor(all_emotion_labels)
# #     )
# #     return at, ae, pt, pe, rt, re, f1t, f1e, cmt, cme


# def select_device(device=""):
#     # referenced from: https://github.com/microsoft/DirectML/blob/master/PyTorch/1.13/classification/train_classification.py#L171
#     if device.lower() == "cuda":
#         if not torch.cuda.is_available():
#             print("torch.cuda not available")
#             return torch.device("cpu")
#         else:
#             return torch.device("cuda:0")
#     if device.lower() == "dml":
#         return td.device(td.default_device())
#     else:
#         return torch.device("cpu")


# class GridSearch:
#     """
#     Creating a grid search for the hyperparameters
#     """

#     def __init__(
#         self,
#         model: Type[MultiTaskModel],
#         train_set,
#         test_set,
#         num_epochs: List[int],
#         batch_size: List[int],
#         learning_rate: List[float],
#         weight_decay: List[float],
#         num_layers: List[int],
#         dropout: List[float],
#         pre_trained_language_model: List[str],
#     ):
#         self.model = model
#         self.train_set = train_set

#         self.test_set = test_set
#         self.weight_decay = weight_decay
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate  # learning rate
#         self.num_layers = num_layers  # number of layers
#         self.dropout = dropout  # dropout rate
#         self.pre_trained_language_model = pre_trained_language_model

#         self.scores = []
#         self.device = select_device("dml")

#     def check_for_checkpoint(self):
#         # Check if there is a checkpoint
#         import json

#         try:
#             with open(f"{BASE_PATH}/logs/scores.json", "r") as f:
#                 checkpoint = json.load(f)

#             if len(checkpoint) > 0:
#                 return checkpoint
#             return None
#         except Exception as e:
#             print("Error in loading the checkpoint", e)
#             return None

#     @staticmethod
#     def find_checkpoint(
#         checkpoint, epochs, batch, weight_decay, lr, layer, drop, pre_train
#     ):
#         for c in checkpoint:
#             if (
#                 c["epochs"] == epochs
#                 and c["batch_size"] == batch
#                 and c["learning_rate"] == lr
#                 and c["num_layers"] == layer
#                 and c["dropout"] == drop
#                 and c["pre_train"] == pre_train
#                 and c["weight_decay"] == weight_decay
#             ):
#                 return True
#         return False

#     def fit(self):
#         """
#         Fits the model
#         """
#         # Check if there is a checkpoint
#         checkpoints = self.check_for_checkpoint()
#         for pre_trained_language_model in self.pre_trained_language_model:
#             if pre_trained_language_model == "bert-base-uncased":
#                 tokenizer = BertTokenizer.from_pretrained(pre_trained_language_model)
#             elif pre_trained_language_model == "roberta-base":
#                 tokenizer = RobertaTokenizer.from_pretrained(pre_trained_language_model)
#             else:
#                 raise ValueError("Please select the right pre-trained language model")
#             for epochs in self.num_epochs:
#                 for batch in self.batch_size:
#                     for weight_decay in self.weight_decay:
#                         for lr in self.learning_rate:
#                             for layer in self.num_layers:
#                                 for drop in self.dropout:
#                                     if self.find_checkpoint(
#                                         checkpoints,
#                                         epochs,
#                                         batch,
#                                         lr,
#                                         weight_decay,
#                                         layer,
#                                         drop,
#                                         pre_trained_language_model,
#                                     ):
#                                         print("Checkpoint found, skipping")
#                                         continue
#                                     print("====================================")
#                                     print(
#                                         "Training the model with the following hyperparameters: "
#                                     )
#                                     print("Epochs: ", epochs)
#                                     print("Batch Size: ", batch)
#                                     print("Learning Rate: ", lr)
#                                     print("Weight Decay: ", weight_decay)
#                                     print("Number of Layers: ", layer)
#                                     print("Dropout: ", drop)
#                                     print(
#                                         "Pre-trained language model: ",
#                                         pre_trained_language_model,
#                                     )
#                                     print("====================================\n\n")

#                                     # Initialize the model
#                                     model = self.model(
#                                         backbone=pre_trained_language_model,
#                                         dropout=drop,
#                                         num_layers=layer,
#                                     )

#                                     criterion = nn.BCEWithLogitsLoss()
#                                     optimizer = optim.SGD(
#                                         model.parameters(),
#                                         lr=lr,
#                                         weight_decay=weight_decay,
#                                     )
#                                     train_set = ValorantChatDataset(
#                                         self.train_set["chat"].values,
#                                         self.train_set["emotion"].values,
#                                         self.train_set["toxicity"].values,
#                                         tokenizer,
#                                     )
#                                     test_set = ValorantChatDataset(
#                                         self.test_set["chat"].values,
#                                         self.test_set["emotion"].values,
#                                         self.test_set["toxicity"].values,
#                                         tokenizer,
#                                     )
#                                     # Data Loaders
#                                     train_dataloader = DataLoader(
#                                         train_set, batch_size=batch, shuffle=True
#                                     )
#                                     test_dataloader = DataLoader(
#                                         test_set, batch_size=batch, shuffle=True
#                                     )
#                                     losses = []
#                                     toxicity_losses = []
#                                     emotion_losses = []

#                                     for epoch in range(1, epochs + 1):
#                                         loss, toxicity_loss, emotion_loss = train_fn(
#                                             model=model,
#                                             criterion=criterion,
#                                             optimizer=optimizer,
#                                             data_loader=train_dataloader,
#                                             device=self.device,
#                                             epoch=epoch,
#                                             total_epoch=epochs,
#                                         )
#                                         losses.append(loss)
#                                         toxicity_losses.append(toxicity_loss)
#                                         emotion_losses.append(emotion_loss)

#                                         print(
#                                             "Loss:",
#                                             loss,
#                                             "Emotion Loss:",
#                                             emotion_loss,
#                                             "Toxicity Loss:",
#                                             toxicity_loss,
#                                             "Epoch:",
#                                             epoch,
#                                         )
#                                     # Evaluate the model on the test set
# #                                     print("Evaluating the model on the test set")
# #                                     (
# #                                         at,
# #                                         ae,
# #                                         pt,
# #                                         pe,
# #                                         rt,
# #                                         re,
# #                                         f1t,
# #                                         f1e,
# #                                         cmt,
# #                                         cme,
# #                                     ) = evaluate_fn(model, test_dataloader, self.device)
# #                                     print(
# #                                         f"""
# # Test Accuracy - Toxicity: {at}
# # Test Accuracy - Emotion: {ae}
# # Test Precision - Toxicity: {pt}
# # Test Precision - Emotion: {pe}
# # Test Recall - Toxicity: {rt}
# # Test Recall - Emotion: {re}
# # Test F1 Score - Toxicity: {f1t}
# # Test F1 Score - Emotion: {f1e}

# # """
#                                     # )

#                                     # print("====================================")
#                                     # print("Saving the model")
#                                     # # Store the scores for this set of hyperparameters
#                                     # self.scores.append(
#                                     #     {
#                                     #         "epochs": epochs,
#                                     #         "batch_size": batch,
#                                     #         "learning_rate": lr,
#                                     #         "weight_decay": weight_decay,
#                                     #         "num_layers": layer,
#                                     #         "dropout": drop,
#                                     #         "pre_train": pre_trained_language_model,
#                                     #         "accuracy_toxicity": at,
#                                     #         "accuracy_emotion": ae,
#                                     #         "loss_average": sum(losses) / len(losses),
#                                     #         "toxicity_loss_average": sum(
#                                     #             toxicity_losses
#                                     #         )
#                                     #         / len(toxicity_losses),
#                                     #         "emotion_loss_average": sum(emotion_losses)
#                                     #         / len(emotion_losses),
#                                     #     }
#                                     # )
#                                     # # Save the scores to a file
#                                     # with open(
#                                     #     f"{BASE_PATH}/logs/scores.json", "w"
#                                     # ) as f:
#                                     #     f.write(str(self.scores))

#                                     # # Save graph of the losses
#                                     # # show_loss_graph(
#                                     # #     losses,
#                                     # #     toxicity_losses,
#                                     # #     emotion_losses,
#                                     # #     f"{epochs}_{batch}_{lr}_{weight_decay}_{layer}_{drop}",
#                                     # # )
#                                     # # show_emotion_loss_graph(
#                                     # #     emotion_losses,
#                                     # #     f"{epochs}_{batch}_{lr}_{weight_decay}_{layer}_{drop}",
#                                     # # )
#                                     # # show_toxicity_loss_graph(
#                                     # #     toxicity_losses,
#                                     # #     f"{epochs}_{batch}_{lr}_{weight_decay}_{layer}_{drop}",
#                                     # # )
#                                     # # Save a checkpoint
#                                     # save_checkpoint(
#                                     #     model.state_dict(),
#                                     #     filename=f"checkpoint_{epochs}_{batch}_{lr}_{weight_decay}_{layer}_{drop}.pt",
#                                     # )


# # def show_loss_graph(losses, toxicity_losses, emotion_losses, title: str):
# #     """
# #     Shows the graph of the losses, save in photo
# #     """
# #     plt.figure(figsize=(10, 5))
# #     plt.plot(losses, label="Total Loss")
# #     plt.plot(toxicity_losses, label="Toxicity Loss")
# #     plt.plot(emotion_losses, label="Emotion Loss")
# #     plt.xlabel("Epochs")
# #     plt.ylabel("Loss")
# #     plt.title(title)
# #     plt.legend()
# #     # Filename <datetime>_loss_graph.png

# #     plt.savefig(
# #         f'{BASE_PATH}/logs/checkpoint_{title}/{datetime.now().strftime("%Y%m%d%H%M%S")}_loss_graph.png'
# #     )
