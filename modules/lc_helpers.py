import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np
import pandas as pd
import pickle
import seaborn as sns


def str_to_number(string):

    result = string

    try:
        result = float(string)
        if result % 1 == 0:
            result = int(result)
    except:
        pass

    return result


def txt_to_dict(filename):
    #  Opens .txt, expecting line format:
    #  key = value

    d = {}
    with open(filename) as f:
        for line in f:
            line = ''.join(line.split())
            line = line.split('#')[0]
            line = line.split('=')
            d[line[0]] = eval(line[1])

    return d

def plot_perc(col, title='', x_label='', y_label='Percentage', filename=False):
    vals = col.unique()
    freqs = [((col == val).sum() / len(col) * 100) for val in vals]

    rcParams['figure.figsize'] = 13, 6.5
    rcParams["axes.labelsize"] = 20
    rcParams['axes.linewidth'] = 1
    rcParams['axes.edgecolor'] = '.5'
    rcParams['legend.frameon'] = False

    fig = sns.barplot(x=vals, y=freqs, palette='Blues_d')
    fig.set_xlabel(x_label, color='white')
    fig.set_ylabel(y_label, color='white')
    fig.set_title(title, color='white', size=30)

    plt.setp(fig.get_xticklabels(), rotation=90, color='white', size=20)
    plt.setp(fig.get_yticklabels(), color='white', size=20)

    if filename != False:
        fig.figure.savefig(filename)

    plt.show()


def plot_dist(df, col, target_col, positives=False):
    distribution = {}
    x_max = df[col].max()

    rcParams['figure.figsize'] = 14.5, 6.5
    rcParams['axes.labelsize'] = 20
    rcParams['axes.linewidth'] = 1
    rcParams['axes.edgecolor'] = '.5'
    rcParams['legend.frameon'] = False

    if positives is False:
        target_vals = df[target_col].unique()
        for n in range(len(target_vals)):
            val = target_vals[n]
            distribution[val] = df[df[target_col] == val][col]
    else:
        distribution['positives'] = df[df[target_col].isin(positives)][col]
        distribution['negatives'] = df[~df[target_col].isin(positives)][col]

    for k in distribution.keys():
       fig = sns.kdeplot(distribution[k], shade=True, bw=(x_max/75), label=f'{target_col} = {k}')

    fig.set_title(f'Distribution of {col}', color='white', size=30)
    fig.tick_params(colors='white')

    plt.setp(fig.get_xticklabels(), color='white', size=15)
    plt.setp(fig.get_yticklabels(), color='white', size=15)
    plt.show()


def pickle_or_alt(file, alt=False, verbose=True):
    # Tries to load a file. If unsuccesful, return alt if it is given

    try:
        with open(file, 'rb') as f:
            result = pickle.load(f)
            print(f'Sucessfully opened {file}')
    except:
        result = alt
        print(f'Could not open {file}')
    return result


def pickle_save(obj, file, verbose=True):
    # Savely saves an object to a file

    try:
        with open(file, 'wb') as f:
            result = pickle.dump(obj, f)
            print(f'Sucessfully saved {file}')
        return True
    except:
        print(f'Could not save {file}')
        return False


def title_to_shortlist(title):
    #  Filters out set words ordered by priority, or returns NaN

    number = 1
    words = ['debt', 'home', 'business', 'card', 'medical', 'wedding', 'vacation', 'purchase',
             'freedom', 'payoff', 'loan','personal', 'other']
    result = 'x'

    for n in range(len(words)):
        if result == 'x':
            word = words[n]
            try:
                if title.find(word) != -1:
                    result = word
            except:
                return title
    return result


def digit_from_num(num, index):
    #  Returns the nth digit of an integer

    import numpy as np

    try:
        return int(str(num)[index])
    except:
        return np.nan


def recreate_sets(df, target_cols):
    #  Splits training and testing data

    X_y = df
    X_y_train = X_y[X_y['set'] == 'train']
    X_y_eval = X_y[X_y['set'] == 'eval']
    X_y_test = X_y[X_y['set'] == 'test']
    X_train = X_y_train.drop(target_cols, axis=1).drop('set', axis=1)
    y_train = X_y_train[target_cols]
    X_eval = X_y_eval.drop(target_cols, axis=1).drop('set', axis=1)
    y_eval = X_y_eval[target_cols]
    X_test = X_y_test.drop(target_cols, axis=1).drop('set', axis=1)
    y_test = X_y_test[target_cols]

    return X_train, X_eval, X_test, y_train, y_eval, y_test


def set_version(file):
    #  Tries to set version to the number storing in file + 1. Otherwise 1

    version = pickle_or_alt(file, 0)
    version += 1
    with open(file, 'wb') as f:
        pickle.dump(version, f)
    return version


def datestamp_to_months(datestamp, lowercase=True):
    #  Converts 'Mmm-YYYY' to an int

    month_dict = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
        'Sep', 'Oct', 'Nov', 'Dec']
    if lowercase:
        month_dict = [x.lower() for x in month_dict]

    try:
        months = month_dict.index(datestamp[:3])
        months += (int(datestamp[-4:]) - 2000) * 12
        return int(months)
    except:
        return datestamp


def months_to_datestamp(months, lowercase=True):
    #  Converts int to 'Mmm-YYYY'

    month_dict = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
        'Sep', 'Oct', 'Nov', 'Dec']
    if lowercase:
        month_dict = [x.lower() for x in month_dict]

    try:
        datestamp = month_dict[months % 12]
        datestamp += "-" + str(int(months / 12))
        return datestamp
    except:
        return months


def percent_to_float(percent):
    #  Converts 'NN%' to float

    try:
        return float(percent[:-1]) / 100
    except:
        return percent
