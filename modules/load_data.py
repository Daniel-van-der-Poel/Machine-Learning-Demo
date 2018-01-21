import numpy as np
import pandas as pd


def load_data(size='s'):
    data = {}
    prefix = r'.\data\LoanStats'
    kwargs = {'low_memory': False, 'skiprows': 1}
    data_sets = [['2007' , 2007, 1, '3a.csv'],
                 ['2012' , 2012, 1, '3b.csv'],
                 ['2014' , 2014, 1, '3c.csv'],
                 ['2015' , 2015, 1, '3d.csv'],
                 ['2016q1', 2016, 1, '_2016Q1.csv'],
                 ['2016q2', 2016, 2, '_2016Q2.csv'],
                 ['2016q3', 2016, 3, '_2016Q3.csv'],
                 ['2016q4', 2016, 4, '_2016Q4.csv'],
                 ['2017q1', 2017, 1, '_2017Q1.csv'],
                 ['2017q2', 2017, 2, '_2017Q2.csv'],
                 ['2017q3', 2017, 3, '_2017Q3.csv']]
    data_sizes = {'s': [4],
                  'm': [4, 8],
                  'l': [0, 2, 4, 6, 9],
                  'xl': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    print(f'Loading dataset{"s" if len(data_sizes[size]) > 1 else ""} ', end='')
    for s in data_sizes[size]:
        set_name = data_sets[s][0]
        year = data_sets[s][1]
        q = data_sets[s][2]
        raw = pd.read_csv(prefix + data_sets[s][3], low_memory=False, skiprows=[0])
        raw = raw.iloc[0:-2, :]
        data[set_name] = {'year': year, 'q': q, 'raw': raw}
        print(f'{set_name} ', end='')

    return data
