from pathlib import Path
# from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import configparser


BASE_PATH = Path(__file__).parent.parent.parent
"""`multi-task model/`"""

# Hyperparameters
# learning_rate = [2e-5, 3e-5, 1e-4]
# weight_decay = [1e-4, 1e-3, 1e-2, 1e-1, 5e-5, 5e-4]
# batch_size = [32, 64]
# num_epochs = [50, 100]
# num_layers = [(64, 32), (128, 64), (256, 128)]
# dropout = [0.2, 0.3, 0.4, 0.5]
# dropout_bert = [0.2, 0.3, 0.4, 0.5]
# pre_trained_language_model = ["bert-base-uncased"]


# Metrics for Pytorch =========================================================
# accuracy_toxicity = Accuracy(task="multiclass", num_classes=17)
# accuracy_emotion = Accuracy(task="multiclass", num_classes=3)

# precision_toxicity = Precision(task="multiclass", average="macro", num_classes=17)
# precision_emotion = Precision(task="multiclass", average="macro", num_classes=3)

# recall_toxicity = Recall(task="multiclass", average="macro", num_classes=17)
# recall_emotion = Recall(task="multiclass", average="macro", num_classes=3)

# f1_toxicity = F1Score(task="multiclass", num_classes=17)
# f1_emotion = F1Score(task="multiclass", num_classes=3)

# confusion_matrix_toxicity = ConfusionMatrix(task="multiclass", num_classes=17)
# confusion_matrix_emotion = ConfusionMatrix(task="multiclass", num_classes=3)

# AUROC_toxicity = AUROC(task="multiclass", num_classes=17)
# AUROC_emotion = AUROC(task="multiclass", num_classes=3)
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

def get_parameters_from_config(config_file='default.ini'):
    """
    Get hyperparameters from a configuration file.

    Parameters:
    - config_file (str): Path to the configuration file.

    Returns:
    - dict: Dictionary containing hyperparameters.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    hyperparameters = {}
    try:
        #   epochs = 50
        # num_layers = 64
        # learning_rate = 1e-4
        # batch_size = 32
        # dropout = 0.5
        # weight_decay = 0.1
        # l2_reg_emotion = 0.03
        # l2_reg_toxicity = 0.025 
        #
        hyperparameters['epochs'] = config.getint('Hyperparameters', 'epochs')
        hyperparameters['num_layers'] = config.getint('Hyperparameters', 'num_layers')
        hyperparameters['learning_rate'] = config.getfloat('Hyperparameters', 'learning_rate')
        hyperparameters['batch_size'] = config.getint('Hyperparameters', 'batch_size')
        hyperparameters['dropout'] = config.getfloat('Hyperparameters', 'dropout')
        hyperparameters['learning_decay'] = config.getfloat('Hyperparameters', 'learning_decay')
        hyperparameters['l2_reg_emotion'] = config.getfloat('Hyperparameters', 'l2_reg_emotion')
        hyperparameters['l2_reg_toxicity'] = config.getfloat('Hyperparameters', 'l2_reg_toxicity')
        hyperparameters['l2_reg_lstm'] = config.getfloat('Hyperparameters', 'l2_reg_lstm')
        hyperparameters['weight_epoch'] = config.getint('Hyperparameters', 'weight_epoch')
        hyperparameters['k_fold'] = config.getint('Hyperparameters', 'k_fold')
    except configparser.Error as e:
        raise ValueError(f"Error reading hyperparameters from config file: {e}")

    return hyperparameters

def get_gs_hyperparameters_from_config(config_file='default.ini'):
    """
    Get Grid Search hyperparameters from a configuration file.

    Parameters:
    - config_file (str): Path to the configuration file.

    Returns:
    - dict: Dictionary containing hyperparameters.
    """

    config = configparser.ConfigParser()
    config.read(config_file)

    hyperparameters = {}
    try:
        hyperparameters['learning_rate'] = config.get('GridSearch', 'learning_rate')
        hyperparameters['batch_size'] = config.get('GridSearch', 'batch_size')
        hyperparameters['dropout'] = config.get('GridSearch', 'dropout')
        hyperparameters['num_layers'] = config.get('GridSearch', 'num_layers')
        hyperparameters['epochs'] = config.get('GridSearch', 'epochs')
        hyperparameters['learning_decay'] = config.get('GridSearch', 'learning_decay')
        hyperparameters['l2_reg_emotion'] = config.get('GridSearch', 'l2_reg_emotion')
        hyperparameters['l2_reg_toxicity'] = config.get('GridSearch', 'l2_reg_toxicity')
        hyperparameters['l2_reg_lstm'] = config.get('GridSearch', 'l2_reg_lstm')
        hyperparameters['weight_epoch'] = config.get('GridSearch', 'weight_epoch')
    except configparser.Error as e:
        raise ValueError(f"Error reading hyperparameters from config file: {e}")

    return hyperparameters