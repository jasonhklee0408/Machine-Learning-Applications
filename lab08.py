import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def best_transformation():
    """
    Returns an integer corresponding to the correct option.

    :Example:
    >>> best_transformation() in [1,2,3,4]
    True
    """

    # take log and square root of the dataset
    # look at the fit of the regression line (and R^2)

    return 1

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------


def create_ordinal(df):
    """
    create_ordinal takes in diamonds and returns a dataframe of ordinal
    features with names ordinal_<col> where <col> is the original
    categorical column name.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_ordinal(diamonds)
    >>> set(out.columns) == {'ordinal_cut', 'ordinal_clarity', 'ordinal_color'}
    True
    >>> np.unique(out['ordinal_cut']).tolist() == [0, 1, 2, 3, 4]
    True
    """
    cat_cols = ['cut', 'color', 'clarity']
    output = pd.DataFrame()
    def col_mapping(col, output):
        ranks = list(df[col].unique())[::-1]
        transform = df[col].apply(lambda x: ranks.index(x))
        output['ordinal_'+col] = transform

    for column in cat_cols:
        col_mapping(column, output)

    return output

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def create_one_hot(df):
    """
    create_one_hot takes in diamonds and returns a dataframe of one-hot
    encoded features with names one_hot_<col>_<val> where <col> is the
    original categorical column name, and <val> is the value found in
    the categorical column <col>.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_one_hot(diamonds)
    >>> out.shape == (53940, 20)
    True
    >>> out.columns.str.startswith('one_hot').all()
    True
    >>> out.isin([0,1]).all().all()
    True
    """
    cat_cols = ['cut', 'color', 'clarity']
    output = pd.DataFrame()

    def one_hot_helper(col):
        out = pd.DataFrame()
        vals = list(col.cat.categories)
        for val in vals:
            col_name = 'one_hot_' + col.name + '_' + val
            out[col_name] = col.apply(lambda x: 1 if x == val else 0)
        return out

    for i in cat_cols:
        cat_dataframe = one_hot_helper(df[i])
        output = pd.concat([output, cat_dataframe], axis = 1)
    return output


def create_proportions(df):
    """
    create_proportions takes in diamonds and returns a
    dataframe of proportion-encoded features with names
    proportion_<col> where <col> is the original
    categorical column name.

    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_proportions(diamonds)
    >>> out.shape[1] == 3
    True
    >>> out.columns.str.startswith('proportion_').all()
    True
    >>> ((out >= 0) & (out <= 1)).all().all()
    True
    """
    cat_cols = ['cut', 'color', 'clarity']
    output = pd.DataFrame()
    for i in cat_cols:
        vals = df[i].value_counts()
        vals = vals/sum(vals)
        output['proportion_' + i] = df[i].apply(lambda x: vals.loc[x]).astype(float)
    return output

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def create_quadratics(df):
    """
    create_quadratics that takes in diamonds and returns a dataframe
    of quadratic-encoded features <col1> * <col2> where <col1> and <col2>
    are the original quantitative columns
    (col1 and col2 should be distinct columns).

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_quadratics(diamonds)
    >>> out.columns.str.contains(' * ').all()
    True
    >>> ('x * z' in out.columns) or ('z * x' in out.columns)
    True
    >>> out.shape[1] == 15
    True
    """
    df = df[['carat','depth','table','x','y','z']]
    output = pd.DataFrame()
    columns = list(df.columns)
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            output[columns[i]+' * ' + columns[j]] = df[columns[i]] * df[columns[j]]
    return output


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def comparing_performance():
    """
    Hard coded answers to comparing_performance.

    :Example:
    >>> out = comparing_performance()
    >>> len(out) == 6
    True
    >>> import numbers
    >>> isinstance(out[0], numbers.Real)
    True
    >>> all(isinstance(x, str) for x in out[2:-1])
    True
    >>> 0 <= out[-1] <= 1
    True
    """

    # create a model per variable => (variable, R^2, RMSE) table

    return [0.8493305264354858, 1548.5331930613174, 'x', 'carat * x', 'color', 0.04143165553562414]

# ---------------------------------------------------------------------
# Question # 6, 7, 8
# ---------------------------------------------------------------------


class TransformDiamonds(object):

    def __init__(self, diamonds):
        self.data = diamonds

    def transformCarat(self, data):
        """
        transformCarat takes in a dataframe like diamonds
        and returns a binarized carat column (an np.ndarray).

        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transformCarat(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> transformed[172, 0] == 1
        True
        >>> transformed[0, 0] == 0
        True
        """
        vals = self.data['carat'].to_numpy().reshape(-1, 1)
        bi = Binarizer(1)
        binarized = bi.transform(vals)
        return binarized

    def transform_to_quantile(self, data):
        """
        transform_to_quantiles takes in a dataframe like diamonds
        and returns an np.ndarray of quantiles of the weight
        (i.e. carats) of each diamond.

        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds.head(10))
        >>> transformed = out.transform_to_quantile(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> 0.2 <= transformed[0,0] <= 0.5
        True
        >>> np.isclose(transformed[1,0], 0, atol=1e-06)
        True
        """
        vals = self.data['carat'].to_numpy().reshape(-1, 1)
        quantile = QuantileTransformer()
        quantile.fit(vals)
        return quantile.transform(vals)

    def transform_to_depth_pct(self, data):
        """
        transform_to_volume takes in a dataframe like diamonds
        and returns an np.ndarray consisting of the approximate
        depth percentage of each diamond.

        :Example:
        >>> diamonds = sns.load_dataset('diamonds').drop(columns='depth')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transform_to_depth_pct(diamonds)
        >>> len(transformed.shape) == 1
        True
        >>> np.isclose(transformed[0], 61.286, atol=0.0001)
        True
        """
        x = self.data['x'].to_numpy()
        y = self.data['y'].to_numpy()
        z = self.data['z'].to_numpy()
        def depth_percentage(input):
            denominator = (input[0]+input[1])/2
            numerator = input[2]*100
            return numerator/denominator

        func = FunctionTransformer(depth_percentage)
        return func.transform([x,y,z])


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['best_transformation'],
    'q02': ['create_ordinal'],
    'q03': ['create_one_hot', 'create_proportions'],
    'q04': ['create_quadratics'],
    'q05': ['comparing_performance'],
    'q06,7,8': ['TransformDiamonds']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """

    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
