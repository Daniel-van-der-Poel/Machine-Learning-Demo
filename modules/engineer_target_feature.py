import numpy as np
import pandas as pd

def engineer_target_feature(input_df, target_cols, positives):
    """
    By default, creates a binary target in which 'late' and 'default' are 1s and the rest 0s.
    Supports multiclass target features for future use.
    """

    print('...\nEngineering target feature')

    df = input_df

    for col in target_cols:
        df[col] = df[col].apply(
            lambda x: 1 if x in positives else 0)

    n_training = df[df['set'] == 'train'].shape[0]
    n_eval = df[df['set'] == 'eval'].shape[0]
    n_test = df[df['set'] == 'test'].shape[0]
    n_features = df.shape[1] - 2 #  -2 because of target feature and 'set'
    n_vals = df[target_cols].shape[0]
    perc_1 = float(df[target_cols].sum() / n_vals)
    perc_0 = float((n_vals - df[target_cols].sum()) / n_vals)

    print(f'Final number of samples: {n_training} training, {n_eval} cross-validation, {n_test} test')
    print(f'Final number of features: {n_features}')
    print('Target feature: {:.3%} 0s, {:.3%} 1s'.format(perc_0, perc_1))

    return df, target_cols
