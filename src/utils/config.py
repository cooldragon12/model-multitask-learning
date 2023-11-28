from pathlib import Path
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, ConfusionMatrix

BASE_PATH = Path(__file__).parent.parent.parent
"""`multi-task model/`"""

# Hyperparameters
learning_rate = [0.01, 0.1, 0.15]
weight_decay = [1e-4, 1e-3, 1e-2, 1e-1, 5e-5,5e-4]
batch_size = [16, 32]
num_epochs = [50, 100, 200]
num_layers = [1, 2, 3]
dropout = [0.2, 0.3, 0.4, 0.5]
pre_trained_language_model = ['bert-base-uncased', 'roberta-base']


# Metrics
accuracy_toxicity = Accuracy(task="multiclass", num_classes=17)
accuracy_emotion = Accuracy(task="multiclass", num_classes=3)

precision_toxicity = Precision(task="multiclass",average='macro', num_classes=17)
precision_emotion = Precision(task="multiclass", average='macro',num_classes=3)

recall_toxicity = Recall(task="multiclass", average='macro',num_classes=17)
recall_emotion = Recall(task="multiclass", average='macro',num_classes=3)

f1_toxicity = F1Score(task="multiclass",num_classes=17)
f1_emotion = F1Score(task="multiclass",num_classes=3)

confusion_matrix_toxicity = ConfusionMatrix(task="multiclass",num_classes=17)
confusion_matrix_emotion = ConfusionMatrix(task="multiclass",num_classes=3)

AUROC_toxicity = AUROC(task="multiclass",num_classes=17)
AUROC_emotion = AUROC(task="multiclass",num_classes=3)

# Path: src/utils/config.py