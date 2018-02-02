"""
Import libraries
"""

import datetime
import numpy as np
import pandas as pd
from scipy.stats import skewnorm


"""
Define generator functions

Note: generators will return a single value if size == 1, or an array of values if size > 1

"""

def rand_uniform(low=1, high=10, size=1, decimals='all'):
    #  Uniform distribution from low (inclusive) to high (exclusive)
    if decimals == 0:
        data = np.random.randint(low, high, size=size)
    elif decimals == 'all':
        data = np.random.uniform(low, high, size)
    else:
        data = np.around(np.random.uniform(low, high, size=size), decimals)

    if size == 1:
        return data[0]
    else:
        return data


def rand_exponential(scale=1, loc=0, size=1, reverse=False, decimals='all'):
    #  Exponential distribution starting at loc
    data = np.random.exponential(scale=scale, size=size)

    if reverse:
        data *= -1

    data += loc

    if decimals != 'all':
        data = np.around(data, decimals)
    if decimals == 0:
        data = data.astype(int)

    if size == 1:
        return data[0]
    else:
        return data

def rand_normal(loc=0, sigma=1, size=1, skew=0, decimals='all'):
    #  Normal distribution centered at loc
    data = skewnorm.rvs(a=skew, loc=loc, scale=sigma, size=size)

    if decimals != 'all':
        data = np.around(data, decimals)
    if decimals == 0:
        data = data.astype(int)

    if size == 1:
        return data[0]
    else:
        return data


def rand_triangular(low=1, mode=4.5, high=10, size=1, decimals='all'):
    #  Triangular distribution (mode is the 'peak')
    data = np.random.triangular(low, mode, high, size=size)

    if decimals != 'all':
        data = np.around(data, decimals)
    if decimals == 0:
        data = data.astype(int)

    if size == 1:
        return data[0]
    else:
        return data


def rand_log(low=0, high=10, size=1, reverse=False, decimals='all'):
    #  Logaritmic distribution from low to high (inclusive)
    data = pd.Series(rand_uniform(low=10, high=100, size=size))
    data = np.log10(data) - 1

    if not reverse:
        data = 1 - data
    data *= (high - low) #  range [0, high-low]
    data += low #  range [low, high]

    if decimals != 'all':
        data = np.around(data, decimals)
    if decimals == 0:
        data = data.astype(int)

    if size == 1:
        return list(data)[0]
    else:
        return list(data)


def rand_choice(classes, size=1):
    #  Choice from an array of classes with equal probability
    data = np.random.choice(classes, size=size)

    if size == 1:
        return data[0]
    else:
        return data


def rand_weighted_choice(classes, size=1):
    #  Choice from a dictionary of classes with weighted probability
    #  Format: 'class': p

    pvals = []

    for c in classes:
        pvals.append(classes[c])

    data =  np.random.choice(list(classes.keys()), p=pvals, size=size)

    if size == 1:
        return data[0]
    else:
        return data


def rand_date(low, high, size=1, past=True, future=True, as_string=False):
    #  Date or date string (YYYY-MM-DD) between dates low and high. Precision is 1 week

    now = datetime.datetime.now()
    low -= now.year
    high = high - now.year
    weeks_low = low * 52 - 52
    weeks_high = high * 52

    data = pd.Series([now + datetime.timedelta(weeks=np.random.randint(weeks_low,weeks_high)) for x in range(0, size)])

    if past is False:
        data = data.apply(lambda x: now if x < now else x)
    if future is False:
        data = data.apply(lambda x: now if x > now else x)
    if as_string:
        data = data.apply(lambda x: f'{x.year}-{x.month:02d}-{x.day:02d}')

    if size == 1:
        return list(data)[0]
    else:
        return list(data)
