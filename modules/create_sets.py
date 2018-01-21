import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_sets(df, id_col, target_cols, eval_size=0.2, test_size=.2, datestamp='datestamp'):

    print('Preparing data')

    #  Remove to empty columns and convert datestamp to month integers
    for dataset in df:
        df[dataset]['prep'] = df[dataset]['raw'].iloc[:, 2:]
        df[dataset]['prep'][datestamp] = df[dataset]['year'] * 12 + (df[dataset]['q'] - 1) * 3

    #  Merge all datasets into one
    print(f'Splitting dataset{"s" if len(df.keys()) > 1 else ""} ', end='')
    [print(f'{key} ', end='') for key in df.keys()], print()
    X_y = pd.concat([df[key]['prep'] for key in df.keys()])

    #  Drop samples with no target value
    n_dropped = X_y[X_y[target_cols].isnull().any(axis=1)].shape[0]
    print(f'Dropping {n_dropped:,} samples without a target value')
    X_y = X_y[X_y[target_cols].notnull().any(axis=1)]

    #  Create training and testing datasets
    X = X_y.drop(target_cols, axis=1)
    y = X_y[target_cols]
    eval_size /= 1 - test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                        random_state=1)

    X_train['set'] = 'train'
    X_eval['set'] = 'eval'
    X_test['set'] = 'test'

    #  Rebuild and return X_y
    X_y_train = pd.concat([X_train, y_train], axis=1)
    X_y_eval = pd.concat([X_eval, y_eval], axis=1)
    X_y_test = pd.concat([X_test, y_test], axis=1)
    X_y = pd.concat([X_y_train, X_y_eval, X_y_test], ignore_index=True)

    print('Initial number of samples: {:,} training, {:,} cross-validation, {:,} test'.format(
        X_y[X_y['set'] == 'train'].shape[0], X_y[X_y['set'] == 'eval'].shape[0],
        X_y[X_y['set'] == 'test'].shape[0]))
    print(f'Initial number of features: {X_y.drop(target_cols, axis=1).shape[1]:,}')

    return X_y
