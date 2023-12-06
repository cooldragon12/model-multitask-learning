from pathlib import Path
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, ConfusionMatrix

BASE_PATH = Path(__file__).parent.parent.parent
"""`multi-task model/`"""

# Hyperparameters
learning_rate = [2e-5, 3e-5, 1e-4]
weight_decay = [1e-4, 1e-3, 1e-2, 1e-1, 5e-5, 5e-4]
batch_size = [32, 64]
num_epochs = [50, 100]
num_layers = [(64, 32), (128, 64), (256, 128)]
dropout = [0.2, 0.3, 0.4, 0.5]
dropout_bert = [0.2, 0.3, 0.4, 0.5]
pre_trained_language_model = ["bert-base-uncased"]


# Metrics for Pytorch =========================================================
accuracy_toxicity = Accuracy(task="multiclass", num_classes=17)
accuracy_emotion = Accuracy(task="multiclass", num_classes=3)

precision_toxicity = Precision(task="multiclass", average="macro", num_classes=17)
precision_emotion = Precision(task="multiclass", average="macro", num_classes=3)

recall_toxicity = Recall(task="multiclass", average="macro", num_classes=17)
recall_emotion = Recall(task="multiclass", average="macro", num_classes=3)

f1_toxicity = F1Score(task="multiclass", num_classes=17)
f1_emotion = F1Score(task="multiclass", num_classes=3)

confusion_matrix_toxicity = ConfusionMatrix(task="multiclass", num_classes=17)
confusion_matrix_emotion = ConfusionMatrix(task="multiclass", num_classes=3)

AUROC_toxicity = AUROC(task="multiclass", num_classes=17)
AUROC_emotion = AUROC(task="multiclass", num_classes=3)
# =============================================================================


# Use on Path: src/build/tensorflow_model/preprocessing.py
# Use on Path: src/build/tensorflow_model/model.py
# Use on Path: src/build/pytorch/model.py
# Use on Path: src/build/pytorch/preprocessing.py
EMOTIONS_LABELS = ["positive", "negative", "neutral"]
TOXICITY_LABELS = [
    "not toxic",
    "cyberbullying",
    "blaming others",
    "sarcasm", 
    "rng complaints",
    "gamesplaining",
    "sexism",
    "ebr complaints",
    "mm complaints",
    "male preserve",
    "racism",
    "ableism",
    "ageism",
    "game complaints"
]
