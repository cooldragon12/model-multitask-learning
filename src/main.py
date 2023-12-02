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
            elif not getattr(args[0], 'learning_rate', None) and not getattr(args[0], 'num_layers', None) and getattr(args[0], 'dropout', None) and not getattr(args[0], 'epochs', None) and not getattr(args[0], 'batch_size', None):
                print("Running the model with custom hyperparameters")
                try:
                    train_with_hyperparameter(args[0].epochs, args[0].batch_size, args[0].learning_rate, args[0].num_layers, args[0].dropout)
                except Exception as e:
                    raise Exception("Error in running the hyperparameter:", e)
            else:
                print("Starting training with default hyperparameters")
                try:
                    train_with_hyperparameter(30, 64, 1e-4, 64, 0.3)
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
    # if args[0].train:
    #     if args[0].grid_search:
    #         print('Running the model with grid search\n\n')
    #         try:
    #             run_training_with_grid_search()
    #         except Exception as e:
    #             raise Exception("Error in running the grid search:", e)
    #     elif args[0].learning_rate and args[0].num_layers and args[0].dropout and args[0].epochs and args[0].batch_size:
    #         run_training_with_hyperparameter_tuning(args[0].epochs, args[0].batch_size, args[0].learning_rate, args[0].num_layers, args[0].dropout)
    #     else:
    #         raise ValueError('Please provide the hyperparameters for the model')
    # elif args[0].run:
    #     raise NotImplementedError('Try not implemented yet')
    # elif args[0].evaluate:
    #     raise NotImplementedError('Evaluation not implemented yet')
    # elif args[0].learning_rate and args[0].num_layers and args[0].dropout and args[0].epochs and args[0].batch_size:
    #     print('Running the model with custom hyperparameters')
    # else:
    #     raise ValueError('Please select the right option/flag')
if __name__ == "__main__":
    try:
        args = get_cli_argument()
        main(args)
    except Exception as e:
        print(e)