from utils.config import get_parameters_from_config
from utils.cli import get_cli_argument
from train_tensorflow import grid_search_train, train_with_hyperparameter


def main(*args):
    if args[0].where == 'pytorch':
        raise NotImplementedError('Pytorch not implemented yet')
    elif args[0].where == 'tensorflow':
        if args[0].train:
            if args[0].grid_search:
                print('Running the model with grid search\n\n')
                try:
                    grid_search_train()
                except Exception as e:
                    raise Exception("Error in running the grid search:", e)
            elif any([getattr(args, attr, None) is not None for attr in ['learning_rate', 'num_layers', 'dropout', 'epochs', 'batch_size', 'l2_reg_emotion', 'l2_reg_toxicity', 'weight_decay', 'weight_epoch']]):
                print("Running the model with custom hyperparameters")
                try:
                    train_with_hyperparameter(
                        epochs= args[0].epochs,
                        batch_size=args[0].batch_size,
                        learning_rate=args[0].learning_rate,
                        lstm_layers=args[0].num_layers,
                        dropout=args[0].dropout,
                        weight_epoch=args[0].weight_epoch,
                        l2_emotion=args[0].l2_reg_emotion,
                        l2_toxicity=args[0].l2_reg_toxicity,
                        learning_decay=args[0].learning_decay
                    )
                except Exception as e:
                    raise Exception("Error in running the hyperparameter:", e)
            else:
                print("Starting training with default hyperparameters")
                try:
                    default_hyperparameters = get_parameters_from_config()
                    train_with_hyperparameter(
                        epochs= default_hyperparameters['epochs'],
                        batch_size=default_hyperparameters['batch_size'],
                        learning_rate=default_hyperparameters['learning_rate'],
                        lstm_layers=default_hyperparameters['num_layers'],
                        dropout=default_hyperparameters['dropout'],
                        weight_epoch=default_hyperparameters['weight_decay'],
                        l2_emotion=default_hyperparameters['l2_reg_emotion'],
                        l2_toxicity=default_hyperparameters['l2_reg_toxicity']
                    )
                except Exception as e:
                    raise Exception("Error in running the hyperparameter:", e)
        elif args[0].run:
            raise NotImplementedError('Try not implemented yet')
        elif args[0].evaluate:
            raise NotImplementedError('Evaluation not implemented yet')
        else:
            raise ValueError('Please select the right option/flag')
    else:
        raise ValueError('Please select the right option/flag')
if __name__ == "__main__":
    try:
        args = get_cli_argument()
        main(args)
    except Exception as e:
        print(e)