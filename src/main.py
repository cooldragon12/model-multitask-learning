



def main(*args):
    if args[0].where == 'pytorch':
        raise NotImplementedError('Pytorch not implemented yet')
    elif args[0].where == 'tensorflow':
        if args[0].train:
            from train_tensorflow import grid_search_train, train_with_hyperparameter, k_fold_training_with_current_config
            from utils.config import get_gs_hyperparameters_from_config, get_parameters_from_config
            if args[0].grid_search:
                print('Running the model with grid search\n\n')
                try:
                    gs_hyperparameters = get_gs_hyperparameters_from_config()
                    grid_search_train(
                        epochs_options= gs_hyperparameters['epochs'],
                        batch_size_options=gs_hyperparameters['batch_size'],
                        learning_rate_options=gs_hyperparameters['learning_rate'],
                        lstm_layers_options=gs_hyperparameters['num_layers'],
                        dropouts_options=gs_hyperparameters['dropout'],
                        weight_epoch_options =gs_hyperparameters['weight_epoch'],
                        l2_emotion_options=gs_hyperparameters['l2_reg_emotion'],
                        l2_toxicity_options=gs_hyperparameters['l2_reg_toxicity'],
                        learning_decay_options=gs_hyperparameters['learning_decay'],
                        l2_lstm_options=gs_hyperparameters['l2_reg_lstm']
                    )
                    
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
                        learning_decay=args[0].learning_decay,
                        l2_lstm=args[0].l2_reg_lstm
                    )
                except Exception as e:
                    raise Exception("Error in running the hyperparameter:", e)
            elif args[0].k_fold:
                print("Running the model with k-fold cross validation")
                try:
                    default_hyperparameters = get_parameters_from_config()
                    k_fold_training_with_current_config(
                        epochs= default_hyperparameters['epochs'],
                        batch_size=default_hyperparameters['batch_size'],
                        learning_rate=default_hyperparameters['learning_rate'],
                        lstm_layers=default_hyperparameters['num_layers'],
                        dropout=default_hyperparameters['dropout'],
                        weight_epoch=default_hyperparameters['weight_epoch'],
                        learning_decay=default_hyperparameters['learning_decay'],
                        l2_lstm=default_hyperparameters['l2_reg_lstm'],
                        k_fold=default_hyperparameters['k_fold']

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
                        weight_epoch=default_hyperparameters['weight_epoch'],
                        l2_emotion=default_hyperparameters['l2_reg_emotion'],
                        l2_toxicity=default_hyperparameters['l2_reg_toxicity'],
                        learning_decay=default_hyperparameters['learning_decay'],
                        l2_lstm=default_hyperparameters['l2_reg_lstm']
                    )
                except Exception as e:
                    raise Exception("Error in running the hyperparameter:", e)
        elif args[0].evaluate:
            print("Evaluating the model")
            import tensorflow as tf
            from transformers import TFBertModel
            from utils.utils import encode_texts, Decoder
            custom_objects = {'TFBertModel': TFBertModel}
            decode = Decoder('preprocessed_df_combined.pkl')
            # load the model
            model = tf.keras.models.load_model('bi_lstm_bert_model_30_64_adam_0_6_42_21_l1l2_0_085_0_09.h5', custom_objects=custom_objects)
            while True:
                try:
                    text = input("Enter the text: ")
                    if text == 'exit':
                        break
                    text_encoded = encode_texts(text)
                    print("Text: ",text)
                    pred = model.predict(text_encoded)
                    print("Predictions: ", "\nEmotion:",decode.decode_emotion(pred[0]), "\nToxicity:",decode.decode_toxicity(pred[1]))
                except Exception as e:
                    print(e)
        elif args[0].run:
            print("Trying the model")
            import tensorflow as tf
            from transformers import TFBertModel
            import tensorflowjs as tfjs
            custom_objects = {'TFBertModel': TFBertModel}
            # load the model
            model = tf.keras.models.load_model(f'bi_lstm_bert_model_30_64_adam_0_6_42_21_l1l2_0_085_0_09.h5', custom_objects=custom_objects)

            tfjs.converters.save_keras_model(model, 'bi_lstm_bert_model_30_64_adam_0-6_42_21_l1l2_0-085_0-09')
            model.summary()
        else:
            raise ValueError('Please select the right option/flag')
    else:
        raise ValueError('Please select the right option/flag')
if __name__ == "__main__":
    from utils.cli import get_cli_argument  
    try:
        args = get_cli_argument()
        main(args)
    except Exception as e:
        print(e)