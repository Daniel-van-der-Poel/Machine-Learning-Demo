import numpy as np
import pandas as pd
from ml_helpers import *
from lc_helpers import *


def engineer_features(input_df, target_cols, ordinal_cols, interval_cols, date_cols, unimportant_cols=[],
                      cheat_cols=[], keep_only_cols=False, datestamp='datestamp', max_nan=.4, min_frequency=0.9999,
                      normalise=True, rem_corr=True, max_corr=0.99, deskew=False):

    print('Engineering features')

    df = input_df

    #  Drop sparse training samples
    df[df['set'] == 'train'] = df[df['set'] == 'train'][filter_sparse_samples(df[df['set'] == 'train'], max_nan)]

    #  Make all strings lowercase
    print('Converting all strings to lowercase')
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.lower() if type(x) is str else x)

    #  Convert grade letters to integers
    print('Converting ordinals, binaries and number-like strings to integers')
    df['grade'] = df['grade'].apply(lambda x: (ord(x[0]) - 64) if type(x) is str else x)
    df['sub_grade'] = df['grade'].apply(
        lambda x: (ord(x[0]) - 64) * 10 + int(x[1]) if type(x) is str else x)

    #  Convert various classes to integers
    df['verification_status'] = df['verification_status'].astype(str).replace(
        {'not verified': 0, 'verified': 1, 'source verified': 2})
    df['application_type'].replace({'individual': 0, 'joint app': 1}, inplace=True)
    df['verification_status_joint'] = df['verification_status_joint'].astype(str).replace(
        {np.nan: 1, 'not verified': 0})
    df['hardship_type'] = df['hardship_type'].astype(str).replace(
        {np.nan: 0, 'interest only-3 months deferral': 1})
    df['disbursement_method'].replace({'cash': 0, 'directpay': 1}, inplace=True)
    df['initial_list_status'].replace({'w': 0, 'f': 1}, inplace=True)

    #  Convert y/n to integers
    for col in ['pymnt_plan', 'hardship_flag', 'debt_settlement_flag']:
        df[col].replace({'n': 0, 'y': 1}, inplace=True)

    #  Convert percentage strings to floats
    df['int_rate'] = df['int_rate'].apply(percent_to_float)
    df['revol_util'] = df['revol_util'].apply(percent_to_float)

    #  Convert datestamps to months as integer
    print('Converting datestamps to integers')
    for col in date_cols - set([datestamp]):
        df[col] = df[col].apply(datestamp_to_months)

    #  Convert terms to integers
    df['term'] = df['term'].apply(lambda x: int(x[:3]) if type(x) is str else x)
    df['term'].replace({36: 0, 60: 1}, inplace=True)

    #  Convert years as string to integer
    df['emp_length'].replace({'n/a': np.nan, '10+ years': 10, '< 1 year': 0, '1 year': 1},
                             inplace=True)
    df['emp_length'] = df['emp_length'].apply(
        lambda x: x[0] if type(x) is str else x).astype('float32')

    #  Split zipcode into separate digits
    print('Converting zipcodes to triple integers')
    df['zip_code'] = df['zip_code'].apply(extract_int_from_str)
    for n in range(3):
        df['zip_d' + str(n)] = df['zip_code'].apply(lambda x: digit_from_num(x, n))
    del df['zip_code']

    #  Calculate description and job title length
    print('Converting descriptions and job titles to string length integers')
    df['emp_title'] = df['emp_title'].str.len()
    df['desc'] = df['desc'].str.len()

    #  Convert 'title' to shortlist
    print('Converting title to shortlist')
    df['title'] = df['title'].apply(title_to_shortlist)

    #  Drop unimportant features
    def drop_unimportant_cols(df, unimportant_cols):
        if unimportant_cols != False:
            unimportant_cols = set(unimportant_cols) & set(df.columns)
            if len(unimportant_cols) > 0:
                print('Dropping {:,} unimportant feature{}'.format(len(unimportant_cols), 's' if len(unimportant_cols) != 1 else ''))
                df.drop(list(unimportant_cols), axis=1, inplace=True)
            return df
        else:
            return df

    df = drop_unimportant_cols(df, unimportant_cols)

    #  Drop 'cheat' features
    def drop_cheat_cols(df, cheat_cols):
        if cheat_cols != False:
            cheat_cols = set(cheat_cols) & set(df.columns)
            if len(cheat_cols) > 0:
                print('Dropping {:,} cheat feature{}'.format(len(cheat_cols), 's' if len(cheat_cols) != 1 else ''))
                df.drop(list(cheat_cols), axis=1, inplace=True)
            return df
        else:
            return df

    df = drop_cheat_cols(df, cheat_cols)

    #  Create dummies for 'object' columns
    object_cols = set(df.select_dtypes(include=['object']).columns) - set(target_cols) - set(['set'])
    df, dummy_cols = dummify(df, object_cols, verbose=1)

    #  Again, drop unimportant features (if a file with a list is supplied)
    df = drop_unimportant_cols(df, unimportant_cols)

    #  And, again, drop 'cheat' features
    df = drop_cheat_cols(df, cheat_cols)

    #  Drop imbalanced columns
    min_frequency = min_frequency
    zero_cols = frequency_selection(df, min_frequency, verbose=False)
    print('Dropping {:,} imbalanced feature{} ({:.2%} frequency threshold)'.format(
        len(zero_cols), 's' if len(zero_cols) != 1 else '', min_frequency))
    df.drop(zero_cols, axis=1, inplace=True)

    #  Update column sets
    ordinal_cols &= set(df.columns)
    interval_cols &= set(df.columns)
    numerical_cols = list(ordinal_cols | interval_cols)

    #  Normalise values
    if normalise:
        print('Normalising values')
        df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].min()) / (df[numerical_cols].max() - df[numerical_cols].min())

    #  Remove intercorrelated features
    if rem_corr:
        drop_cols = find_correlation(df[df['set'] == 'train'], max_corr)
        print('Removing {:,} feature{} with intercorrelation {:.3%} or more'.format(
            len(drop_cols), 's' if len(drop_cols) != 1 else '', max_corr))
        df.drop(drop_cols, axis=1, inplace=True)

    #  Deskew values
    if deskew:
        for col in numerical_cols:
            df[col] = deskew(df[col], threshold=3)

    #  Drop everything but 'keep only' features
    if keep_only_cols != False:
        keep_only_cols = set(keep_only_cols) | set(target_cols) | set(['set'])
        del_cols = set(df.columns) - keep_only_cols
        print('Dropping {:,} feature{}'.format(len(del_cols),
                                             's' if len(del_cols) != 1 else ''))
        df.drop(del_cols, axis=1, inplace=True)

    return df
