import argparse

def get_cli_argument():
    """
    Gets the command line arguments for the MultiTask model.
    """
    parser = argparse.ArgumentParser(description='MultiTask model - Train, Try, Evaluate')
    
    parser.add_argument('-w', '--where', choices=['pytorch', 'tensorflow'], default='tensorflow', help='Where to run the model (default: tensorflow)')
    parser.add_argument('-t', '--train', action='store_true', help='Train the model')
    parser.add_argument('-dp', '--data-path', default='dataset/preprocessed_df.pkl', help='Path to the preprocessed dataframe')
    parser.add_argument('-e', '--evaluate', action='store_true',default=False, help='Evaluate the model')
    parser.add_argument('-r', '--run', action='store_true', help='Try the model')
    parser.add_argument('-gs', '--grid-search', action='store_true', help='Run the grid search using the hyperparameter from: src/utils/config.py')
    parser.add_argument('-kf', '--k-fold', action='store_true', help='Run train with config default and k-fold it: src/utils/config.py')

    sub_custom_parser = parser.add_subparsers(title='Custom Settings', help='Run the model with custom hyperparameters')
    hyperparameter_args = sub_custom_parser.add_parser('custom', help='Run the model with custom hyperparameters')
    hyperparameter_args.add_argument('-lr', '--learning-rate', default=1e-4, type=float, help='Learning rate (default: 0.001)')
    hyperparameter_args.add_argument('-wd', '--learning-decay', default=0, type=float, help='Weight decay (default: 0)')
    hyperparameter_args.add_argument('-nl', '--num-layers', default=64, type=int, help='Number of layers (default: 1)')
    hyperparameter_args.add_argument('-do', '--dropout', default=0.5, type=float, help='Dropout (default: 0.6)')
    hyperparameter_args.add_argument('-ep', '--epochs', default=30, type=int, help='Number of epochs (default: 50)')
    hyperparameter_args.add_argument('-bs', '--batch-size', default=32, type=int, help='Batch size (default: 32)')
    hyperparameter_args.add_argument('-l2e', '--l2-reg-emotion', default=0.03, type=float, help='L2 regularization for emotion (default: 0.03)')
    hyperparameter_args.add_argument('-l2t', '--l2-reg-toxicity', default=0.025, type=float, help='L2 regularization for toxicity (default: 0.025)')
    hyperparameter_args.add_argument('-we', '--weight-epoch', default=5, type=int, help='Weight epoch (default: 5)')
    hyperparameter_args.add_argument('-l2l', '--l2-reg-lstm', default=0.01, type=float, help='L2 regularization for lstm (default: 0.01)')
    # Check for custom settings flag
    if 'custom' in parser.parse_known_args()[0]:
        args = parser.parse_args()
    else:
        args, unknown = parser.parse_known_args()
        if args.train or args.run or args.evaluate or args.grid_search:
            # If any of these flags are set, use default hyperparameters
            args = parser.parse_args()
        else:
            # If no relevant flags are set, use default hyperparameters and mark -t as True
            args.train = True

    return args