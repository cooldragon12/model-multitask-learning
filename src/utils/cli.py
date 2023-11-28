import argparse


def get_cli_argument():
    """
    Gets the command line arguments for the MultiTask model.
    """
    parser = argparse.ArgumentParser(description='MultiTask model - Train, Try, Evaluate')
    
    parser.add_argument('-t','--train', action='store_true', help='Train the model')
    parser.add_argument('-d','--data-path', default='dataset/preprocessed_df.pkl', help='Path to the preprocessed dataframe')
    parser.add_argument('-r','--run', action='store_true', help='Try the model')
    parser.add_argument('-e','--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('-gs','--grid-search', action='store_true', help='Run the grid search using the hyperparameter from: src/utils/config.py')
    parser.add_argument('-wc', '--with-checkpoint',default=True, action='store_true', help='Run the model with the checkpoint')
    sub_custom_parser = parser.add_subparsers(title='Custom Settings',help='Run the model with custom hyperparameters')
    hyperparameter_args = sub_custom_parser.add_parser('custom', help='Run the model with custom hyperparameters')
    hyperparameter_args.add_argument('-lr','--learning-rate', default=0.001, type=float, help=f'Learning rate (default: 0.001)')
    hyperparameter_args.add_argument('-wd','--weight-decay', default=0, type=float, help='Weight decay (default: 0)')
    hyperparameter_args.add_argument('-nl','--num-layers', default=1,type=int, help='Number of layers (default: 1)')
    hyperparameter_args.add_argument('-d','--dropout',default=0.6, type=float, help='Dropout (default: 0.6)')
    hyperparameter_args.add_argument('-ep','--epochs',default=50, type=int, help='Number of epochs (default: 50)')
    hyperparameter_args.add_argument('-bs','--batch-size',default=16, type=int, help='Batch size (default: 16)')
    
    args = parser.parse_args()

    return args