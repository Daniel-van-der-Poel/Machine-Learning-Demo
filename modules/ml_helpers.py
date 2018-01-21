import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats as st


def uniques(data, max_length=0, max_shown=10, ascending=False):
    """
    Ranks the unique values in a dataframe's columns

    Args:
        dataframe: Pandas dataframe/series to be analysed
        max_length: maximum display length of unique values (0 = no limit)
        max_shown: maximum number of unique values shown (0 = no limit)
        ascending: show values from least to most common, or vice versa

    Returns:
        A table that decribes for each column in the dataframe:
        * The total number of values
        * The type of the values
        * The number of unique values, exluding NaN entries
        * The unique non-NaN values, ordered by count
        * The number of NaN entries (if any) and their count
    """

    dataframe = pd.DataFrame(data)
    cols = list(dataframe)
    min_shown = 0

    # Determine the maximum number of unique values that will be shown
    # Then create the dataframe
    for col_name in cols:
        min_shown = np.maximum(
            min_shown, len(dataframe[col_name].value_counts(dropna=True)))
    if max_shown is 0 or max_shown is False:
        max_shown = min_shown
    else:
        max_shown = np.minimum(min_shown, max_shown)
    idx_arr = list(range(1, max_shown + 1))
    row_index = ['type', 'count', 'unique', 'NaN'] + idx_arr
    df = pd.DataFrame(index=row_index, columns=cols)

    # Fill the dataframe
    for col_name in cols:
        col = dataframe[col_name]
        count = col.value_counts(dropna=True)
        vals = count.index.tolist()
        if ascending:
            count = count[::-1]
            vals = vals[::-1]
        nans = col.isnull().sum()
        length = len(col)
        number_values_shown = np.minimum(max_shown, len(vals))
        df.at['type', col_name] = col.dtype
        df.at['count', col_name] = len(col)
        df.at['unique', col_name] = len(vals)
        for i in list(range(number_values_shown)):
            val = str(vals[i])
            val_count = count.iloc[i] / length
            if max_length > 0 and len(val) > max_length:
                val = val[:max_length] + u'\u2026'
            df.at[i + 1, col_name] = ('{}<br>{}<br>{:.1%}'.format(
                val, count.iloc[i], val_count))
        if nans > 0:
            df.at['NaN', col_name] = ('{}<br>{:.1%}'.format(
                nans, nans / length))
        else:
            df.at['NaN', col_name] = ""
    return df.fillna('').style.set_properties(
        **{'word-wrap': 'break-word', 'line-height': '125%'})


def outliers(sr, threshold=5, robust=False, verbose=True):
    """
    Finds the outliers in a Pandas series and returns a list with their indices.

    Args:
        sr: Pandas series
        threshold: maximum deviation to be tolerated
        robust: use median absolute deviation instead of standard deviation
        verbose: print the number of columns dropped

    Returns:
        A list with the indices of the outliers
    """

    method = 'MAD/MeanAd' if robust else 'SD'
    x = sr
    try:
        if robust:
            x = (x - x.median()).abs()  # Absolute distances from the mean
            mad = x.median()  # Median distance from the mean
            if mad == 0:
                x /= 1.486 * x.mean()
            else:
                x /= 1.253314 * mad
        else:
            x = (x - x.mean()).abs() / x.std()
    except:
        print('Found no outliers in {}'.format(sr))
    mask = x > threshold
    if verbose: print('Found {:,} outliers ({} > {}) in {}'.format(
        len(x[mask]), method, threshold, sr.name))
    return sr[mask].index


def polytrans(x, y, order=2):
    """
    Fits values in Pandas series x to the values in series y using an
    nth-order polynomial.

    Args:
        x: Pandas series to be fitted
        y: Pandas series against which x will be fitted
        order: the maximum order polynomial used in the fitting

    Returns:
        The transformed Pandas series x
    """

    def transform_x(x, coef):
        result = .0
        for c in range(len(coef)):
            result += coef[c] * x**(len(coef) - c - 1)
        return result

    data = pd.concat([x, y], axis=1)
    data.dropna(axis=0, how='any', inplace=True)
    try:
        coef = np.polyfit(data[x.name], data[y.name], deg=order)
    except:
        print('Could not polyfit {}'.format(x.name))
    try:
        return x.apply(lambda x: transform_x(x, coef))
    except:
        print('Could not polytrans {}'.format(x.name))


def dummify(input_df, cols='all', dummy_na=False, verbose=False):
    """
    Creates dummy values for selected columns in a Pandas dataframe.

    Args:
        df: Pandas dataframe
        cols: column or list of columns to be 'dummified'. If set to 'All',
            all columns with dtype 'object' will be dummified.

    Returns:
        A Pandas dataframe with dummy columns
    """

    df = input_df
    n_dummified_cols = 0
    n_dummy_cols = 0
    all_dummy_cols = set([])

    if cols is 'all':
        cols = df.select_dtypes(include=['object']).columns
    if type(cols) is str:
        cols = [cols]

    n_dummified_cols = len(cols)

    for col in cols:
        dummy_cols = pd.get_dummies(df[col], col, drop_first=True, dummy_na=dummy_na)
        n_dummy_cols += dummy_cols.shape[1]
        all_dummy_cols |= set(dummy_cols.columns)

        if verbose > 1:
            print(f'Creating dummies for column {col}')

        for dummy in dummy_cols:
            if verbose > 2:
                print(f'...creating dummy column {dummy}')
            df[dummy] = dummy_cols[dummy]
        del df[col]

    if verbose > 0:
        print(f'{n_dummified_cols:,} feature columns were replaced by {n_dummy_cols:,} dummy columns')

    return df, all_dummy_cols


def correlation(x, y, order=2, verbose=True, suppress_warnings=True):
    """
    Finds the maximum correlation between Pandas series x and y after fitting x to y
    using the polytrans function.
    """

    if suppress_warnings:
        import warnings
        warnings.simplefilter('ignore', np.RankWarning)
    corrs = []
    corrs.append(pd.concat([x, y], axis=1).corr().values[0][1]) # Current correlation
    for i in range(2, order + 1):
        corrs.append(pd.concat([polytrans(x, y, order=i), y], axis=1).corr().values[0][1])
    max_corr = max(corrs)
    corr_gain = abs(max_corr) - abs(corrs[0])
    poly = corrs.index(max_corr) + 1

    if verbose:
        if order > 1:
            print('{:.2%} ({:.2%}, order={}) correlation between {} and {}'.
                  format(corrs[0], max_corr, poly, x.name, y.name))
        else:
            print('{:.2%} correlation between {} and {}'.
                  format(corrs[0], x.name, y.name))

    return [corrs[0], max_corr, corr_gain, poly]


def binary_labels(vals, target, force=True, sort=True, verbose=False, order=1):
    """
    Transforms a multiclass Pandas series x into a bivariate one, binning the
    values in x in such a way as to maximise x's correlation to Pandas series y.
    """

    x = vals.name
    y = target.name
    categories = pd.concat([vals, target], axis=1).groupby(x, as_index=False)[y].mean()
    categories = categories.sort_values(y) if sort else categories.sort_values(x)
    categories = categories[x].values
    cat_dict = {}
    if verbose: print(categories, '(no bins)')
    best_corr = [[], correlation(vals, target, order=order, verbose=verbose)[1]]

    for i in range(1, len(categories)):
        cat0, cat1 = categories[:i], categories[i:]
        temp = vals
        pd.options.mode.chained_assignment = None
        temp = temp.apply(lambda n: 0 if n in cat0 else 1)
        if verbose:
            print('\n{} | {} = {} | {} = {:.2%} | {:.2%}'.format(cat0, cat1,len(cat0), len(cat1),
                                                                 (temp==0).sum()/len(temp),
                                                                 (temp==1).sum()/len(temp)))
        corrs = correlation(temp, target, order=order, verbose=verbose)
        if (abs(corrs[1]) > abs(best_corr[1])) or (force and len(best_corr[0]) == 0):
            best_corr = [cat0, corrs[1]]
            improved_corr = True

    if verbose:
        if len(best_corr[0]) > 0:
            print('\nHighest correlation: {:.2%}'.format(best_corr[1]))
        else:
            print('\nInital correlation is highest: {:.2%}'.format(best_corr[1]))

    if len(best_corr[0]) > 0 or force:
        for c in categories:
            cat_dict[c] = 0 if c in best_corr[0] else 1

    return cat_dict


def label_numeric(df, cols, labels, label_nan=False):
    """
    Maps non-numerical values in a Pandas dataframe/series to numeric labels.

    Args:
        df: Pandas dataframe/series
        cols: list of columns with values to be mapped
        labels: array with values ('labels') to be mapped to numbers (equal to their index)
        label_nan: map NaN values to label 0

    Returns:
        A Pandas dataframe/series in which the 'labelled' values have been mapped to numbers
    """

    if type(cols) is str: cols = [cols]
    if label_nan: labels.insert(0, np.nan)
    for c in cols:
        rank = 0

        try:
            for l in labels:
                df[c] = df[c].replace(l, rank)
                rank += 1
        except:
            print('Could not label \'{}\''.format(c))


def frequency_selection(df, threshold=.9, dropna=True, verbose=False):
    """
    Returns a list of highly imbalanced columns, i.e. columns that
    have a single value whose frequency is highter than the frequency threshold.
    By default, columns in which a single value occurs more than 90 % of the time are dropped.

    Args:
        df: Pandas dataframe/series
        threshold: maximum value frequency to be tolerated
        dropna: ignore NaN values
        verbose: print the number of columns dropped

    Returns:
        A dataframe without the imbalanced columns
        """

    result = []

    for c in df.columns:
        counts = df[c].value_counts(ascending=False, dropna=dropna)
        if len(counts) > 0:
            frequency = counts.iloc[0]/len(df[c])
            if frequency >= threshold:
                if verbose > 1:
                    print(f'column {c} has value {counts.index[0]} with a frequency of {frequency:.2%}')
                result.append(c)
        else:
            if verbose > 1:
                print(f'column {c} has no non-NaN values')
            result.append(c)

    if verbose > 0:
        print('{:,}/{:,} columns have a value with a frequency of {:.2%} or higher.'.
              format(len(result), len(df.columns), threshold))

    return result


def pca_df(df, components=0.999, svd='auto', whiten=True):
    """
    Applies PCA to reduce the dimensions of a Pandas dataframe.

    Args:
        df: Pandas dataframe/series
        components: the target number of dimensions (if integer) OR
            the variance to be retained (if float). Setting components=1.0 will retain all
            variance while still trying to reduce dimensionality
        svd: the solver to be used, see See Scikit Learn documentation for details
        whiten: use a transformation that may improve the accuracy of the estimators
            at the cost of losing some information

    Returns:
        A dataframe with reduced dimensions
    """

    pca = PCA(n_components=components, svd_solver=svd, whiten=whiten)
    pca.fit(df)
    result = pd.DataFrame(pca.transform(df), index=df.index)
    result.columns = ['pca_%i' % i for i in range(len(result.columns))]

    return result


def deskew(sr, threshold=2, verbose=True):
    """
    Returns a Pandas dataframe whose skewed columns are deskewed using a Box-Cox transformation.
    By default, columns with skewness over 2 are transformed.

    Args:
        sr: Pandas series
        threshold: maximum tolerated skewness. Columns that are more skewed will be deskewed.
        verbose: print info on the deskwed columns

    Returns:
        A dataframe without the imbalanced columns
    """

    x = sr
    if abs(x.skew()) > threshold:
        if verbose: print('De-skewing {} ({:.2} skewness)'.format(x.name, x.skew()))
        if x.min() <= 0:
            x -= x.min() - 1
        x = st.boxcox(x)[0]

    return x


def extract_int_from_str(string):
    """
    Tries to extract all digits from a string.

    Args:
        string: a string

    Returns:
        An integer consisting of all the digits in the string, in order,
        or False
    """

    try:
        return ''.join(filter(lambda x: x.isdigit(), string))
    except:
        return string


def filter_sparse_samples(df, max_nan=0.5, verbose=True):
    """
    Removes samples (rows) from a Pandas dataframe whose number of NaN values exceeds the set number.

    Args:
        df: Pandas dataframe
        max_nan: maximum tolerated NaN values. If a float, this is a percentage of the features (columns).
        verbose: print number of dropped samples

    Returns:
        A True/False series indicating the sparse samples
    """

    n_samples, n_features = df.shape

    if type(max_nan) is float:
        max_nan = int(max_nan * n_features)

    max_nan = max(0, min(n_features, max_nan)) # 0 <= max_nan <= n_features
    filter_sparse_samples = df.isnull().sum(axis=1) <= max_nan # drop samples with nan > max_nan
    n_sparse = df.shape[0] - df[filter_sparse_samples].shape[0]

    if verbose:
        print(f'{n_sparse:,} out of {n_samples:,} samples contained more than {max_nan} NaN value{"s" if max_nan > 1 else ""} and were dropped')

    return filter_sparse_samples


def find_correlation(df, threshold=0.9):
    """
    Finds highly correlated features in a Pandas dataframe

    Args:
        df: Pandas DataFrame
        threshold: correlation threshold; one of a pair of features with a
            correlation greater than this value will be added to the list

    Returns:
        select_flat: list of columns to be removed
    """

    corr_mat = df.corr()
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][corr_mat[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]

    return select_flat
