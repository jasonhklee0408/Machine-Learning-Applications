import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def data_load(scores_fp):
    """
    follows different steps to create a dataframe
    :param scores_fp: file name as a string
    :return: a dataframe
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> isinstance(scores, pd.DataFrame)
    True
    >>> list(scores.columns)
    ['attempts', 'highest_score']
    >>> isinstance(scores.index[0], int)
    False
    """
    # a
    output = pd.read_csv(scores_fp)
    output = output[['name', 'tries', 'highest_score','sex']]
    # b
    output = output.drop(['sex'], axis=1)

    # c
    output = output.rename(columns = {'name':'firstname', 'tries':'attempts'})

    # d
    output = output.set_index('firstname')

    return output


def pass_fail(scores):
    """
    modifies the scores dataframe by adding one more column satisfying
    conditions from the write up.
    :param scores: dataframe from the question above
    :return: dataframe with additional column pass
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> isinstance(scores, pd.DataFrame)
    True
    >>> len(scores.columns)
    3
    >>> scores.loc["Julia", "pass"]=='Yes'
    True
    """
    def filter_func (data):
        if data['attempts'] < 3:
            if data['highest_score'] >= 50:
                return 'Yes'
            else:
                return 'No'
        elif data['attempts'] < 6:
            if data['highest_score'] >= 70:
                return 'Yes'
            else:
                return 'No'
        elif data['attempts'] < 10:
            if data['highest_score'] >= 90:
                return 'Yes'
            else:
                return 'No'
        else:
            return 'No'
    scores['pass'] = scores.apply(filter_func, axis = 1)
    return scores



def av_score(scores):
    """
    returns the average score for those students who passed the test.
    :param scores: dataframe from the second question
    :return: average score
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> av = av_score(scores)
    >>> isinstance(av, float)
    True
    >>> 91 < av < 92
    True
    """
    passed = scores['pass'] == 'Yes'
    filtered_scores = scores[passed]
    average = filtered_scores['highest_score'].mean()
    return average



def highest_score_name(scores):
    """
    finds the highest score and people who received it
    :param scores: dataframe from the second question
    :return: dictionary where the key is the highest score and the value(s) is a list of name(s)
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> highest = highest_score_name(scores)
    >>> isinstance(highest, dict)
    True
    >>> len(next(iter(highest.items()))[1])
    3
    """
    max_val = scores['highest_score'].max()
    filtered = scores['highest_score'] == max_val
    filtered_data = scores[filtered]
    result = {max_val: filtered_data.index}
    return result


def idx_dup():
    """
    Answers the question in the write up.
    :return:
    >>> ans = idx_dup()
    >>> isinstance(ans, int)
    True
    >>> 1 <= ans <= 6
    True
    """
    return 6



# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def trick_me():
    """
    Answers the question in the write-up
    :return: a letter
    >>> ans =  trick_me()
    >>> ans == 'A' or ans == 'B' or ans == "C"
    True
    """
    data = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]]
    new = pd.DataFrame(data, columns = ['A','B','C'])
    new.to_csv('tricky_1.csv', encoding='utf-8', index = False)
    tricky_2 = pd.read_csv('tricky_1.csv')
    return 'C'



def reason_dup():
    """
     Answers the question in the write-up
    :return: a letter
    >>> ans =  reason_dup()
    >>> ans == 'A' or ans == 'B' or ans == "C"
    True
    """
    return 'A'



def trick_bool():
    """
     Answers the question in the write-up
    :return: a list with three letters
    >>> ans =  trick_bool()
    >>> isinstance(ans, list)
    True
    >>> isinstance(ans[1], str)
    True

    """
    return ['D', 'J', 'M']

def reason_bool():
    """
    Answers the question in the write-up
    :return: a letter
    >>> ans =  reason_bool()
    >>> ans == 'A' or ans == 'B' or ans == "C" or ans =="D"
    True

    """
    return 'B'


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def change(x):
    """
    Returns 'MISSING' when x is `NaN`,
    Otherwise returns x
    >>> change(1.0) == 1.0
    True
    >>> change(np.NaN) == 'MISSING'
    True
    """
    if np.isnan(x):
        return 'MISSING'
    else:
        return x



def correct_replacement(nans):
    """
    changes all np.NaNs to "Missing"
    :param nans: given dataframe
    :return: modified dataframe
    >>> nans = pd.DataFrame([[0,1,np.NaN], [np.NaN, np.NaN, np.NaN], [1, 2, 3]])
    >>> A = correct_replacement(nans)
    >>> (A.values == 'MISSING').sum() == 4
    True

    """
    out = nans.applymap(lambda x: change(x))
    return out


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def population_stats(df):
    """
    population_stats which takes in a dataframe df
    and returns a dataframe indexed by the columns
    of df, with the following columns:
        - `num_nonnull` contains the number of non-null
          entries in each column,
        - `pct_nonnull` contains the proportion of entries
          in each column that are non-null,
        - `num_distinct` contains the number of distinct
          entries in each column,
        - `pct_distinct` contains the proportion of (non-null)
          entries in each column that are distinct from each other.

    :Example:
    >>> data = np.random.choice(range(10), size=(100, 4))
    >>> df = pd.DataFrame(data, columns='A B C D'.split())
    >>> out = population_stats(df)
    >>> out.index.tolist() == ['A', 'B', 'C', 'D']
    True
    >>> cols = ['num_nonnull', 'pct_nonnull', 'num_distinct', 'pct_distinct']
    >>> out.columns.tolist() == cols
    True
    >>> (out['num_distinct'] <= 10).all()
    True
    >>> (out['pct_nonnull'] == 1.0).all()
    True
    """
    output = []
    for column in df:
        non_null = df[column].count()
        pct = non_null/len(df[column])
        dis_entries = df[column].nunique()
        dis_pct = dis_entries/non_null
        row = [non_null,pct,dis_entries,dis_pct]
        output.append(row)
    result = pd.DataFrame(output, index = df.columns, columns = ['num_nonnull','pct_nonnull','num_distinct','pct_distinct'])
    return result


def most_common(df, N=10):
    """
    `most_common` which takes in a dataframe df and returns
    a dataframe of the N most-common values (and their counts)
    for each column of df.

    :param df: input dataframe.
    :param N: number of most common elements to return (default 10)
.
    :Example:
    >>> data = np.random.choice(range(10), size=(100, 2))
    >>> df = pd.DataFrame(data, columns='A B'.split())
    >>> out = most_common(df, N=3)
    >>> out.index.tolist() == [0, 1, 2]
    True
    >>> out.columns.tolist() == ['A_values', 'A_counts', 'B_values', 'B_counts']
    True
    >>> out['A_values'].isin(range(10)).all()
    True
    """
    y = []
    index = []
    cols = {}
    for column in df.columns:
        length = len(column)
        names = df[column].value_counts().index.tolist()
        if N > length:
            for i in range(N-length):
                names.append(np.NaN)
        y.append(names)
        index.append(column + '_values')
        counter = list(df[column].value_counts())
        if N > length:
            for i in range(N-length):
                counter.append(np.NaN)
        y.append(counter)
        index.append(column + '_counts')
    for i in range(len(index)):
        cols[index[i]] = y[i]
    result = pd.DataFrame(cols).head(N)
    return result


# ---------------------------------------------------------------------
# Question 5
# ---------------------------------------------------------------------


def null_hypoth():
    """
    :Example:
    >>> isinstance(null_hypoth(), list)
    True
    >>> set(null_hypoth()).issubset({1,2,3,4})
    True
    """
    return [1]


def simulate_null():
    """
    :Example:
    >>> pd.Series(simulate_null()).isin([0,1]).all()
    True
    """
    return np.random.choice([0,1], p = [0.99,0.01], size = 300)


def estimate_p_val(N):
    """
    >>> 0 < estimate_p_val(1000) < 0.1
    True
    """
    averages = []
    for i in range(N):
        random_sample = simulate_null()
        averages.append(random_sample.sum())
    observed = 8
    filtered_list = list(filter(lambda x: x >= observed, averages))
    return len(filtered_list) / N


# ---------------------------------------------------------------------
# Question 6
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    """
    `super_hero_powers` takes in a dataframe like
    powers and returns a list with the following three entries:
        - The name of the super-hero with the greatest number of powers.
        - The name of the most common super-power among super-heroes whose names begin with 'M'.
        - The most popular super-power among those with only one super-power.

    :Example:
    >>> fp = os.path.join('data', 'superheroes_powers.csv')
    >>> powers = pd.read_csv(fp)
    >>> out = super_hero_powers(powers)
    >>> isinstance(out, list)
    True
    >>> len(out)
    3
    >>> all([isinstance(x, str) for x in out])
    True
    """
    #['Spectre', 'Super Strength', 'Intelligence']
    greatest_powers = powers.set_index(powers['hero_names']).drop(columns =['hero_names']).sum(axis = 1).sort_values(ascending = False).index.tolist()[0]
    m_filtered = powers[powers['hero_names'].str[0] == 'M'].transpose().drop('hero_names')
    m_filtered['sum'] = m_filtered.sum(axis = 1)
    m_power = m_filtered.sort_values(by=['sum'], ascending = False).index.tolist()[0]


    count = list(powers.set_index(powers['hero_names']).drop(columns =['hero_names']).sum(axis = 1))
    powers['count'] = count
    one_filtered = powers[powers['count'] == 1].drop(columns=['hero_names','count']).transpose()
    one_filtered['sum'] = one_filtered.sum(axis = 1)
    one_popular = one_filtered.sort_values(by=['sum'], ascending = False).index.tolist()[0]
    return [greatest_powers, m_power, one_popular]


# ---------------------------------------------------------------------
# Question 7
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    """
    clean_heroes takes in the dataframe heroes
    and replaces values that are 'null-value'
    place-holders with np.NaN.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = clean_heroes(heroes)
    >>> out['Skin color'].isnull().any()
    True
    >>> out['Weight'].isnull().any()
    True
    """
    output = heroes.replace('-',np.NaN).replace(-99, np.NaN)
    return output


def super_hero_stats():
    """
    Returns a list that answers the questions in the notebook.
    :Example:
    >>> out = super_hero_stats()
    >>> out[0] in ['Marvel Comics', 'DC Comics']
    True
    >>> isinstance(out[1], int)
    True
    >>> isinstance(out[2], str)
    True
    >>> out[3] in ['good', 'bad']
    True
    >>> isinstance(out[4], str)
    True
    >>> 0 <= out[5] <= 1
    True
    """

    return ['Marvel Comics',558,'Groot','bad','Onslaught',0.2860824742268041]

# ---------------------------------------------------------------------
# Question 8
# ---------------------------------------------------------------------


def bhbe_col(heroes):
    """
    `bhbe` ('blond-hair-blue-eyes') returns a boolean
    column that labels super-heroes/villains that
    are blond-haired *and* blue eyed.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = bhbe_col(heroes)
    >>> isinstance(out, pd.Series)
    True
    >>> out.dtype == np.dtype('bool')
    True
    >>> out.sum()
    93
    """
    cleaned = clean_heroes(heroes)
    cleaned['bnbe'] = np.where(cleaned['Hair color'].str.contains('lond') & cleaned['Eye color'].str.contains('blue'), True, False)
    return cleaned['bnbe']


def observed_stat(heroes):
    """
    observed_stat returns the observed test statistic
    for the hypothesis test.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = observed_stat(heroes)
    >>> 0.5 <= out <= 1.0
    True
    """
    cleaned = clean_heroes(heroes)
    cleaned['bnbe'] = np.where(cleaned['Hair color'].str.contains('lond') & cleaned['Eye color'].str.contains('blue'), True, False)
    good_bnbe = len(cleaned.loc[cleaned['bnbe'] == True][cleaned['Alignment'] == 'good'].index)
    pop = len(cleaned.loc[cleaned['bnbe'] == True].index)

    return good_bnbe/pop


def simulate_bhbe_null(n):
    """
    `simulate_bhbe_null` that takes in a number `n`
    that returns a `n` instances of the test statistic
    generated under the null hypothesis.
    You should hard code your simulation parameter
    into the function; the function should *not* read in any data.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = simulate_bhbe_null(10)
    >>> isinstance(out, pd.Series)
    True
    >>> out.shape[0]
    10
    >>> ((0.45 <= out) & (out <= 1)).all()
    True
    """
    good_prob = 0.6757493188010899
    bad_prob = 1 - good_prob
    results = []
    for i in range(n):
        sample = np.random.choice([0,1], p = [bad_prob, good_prob], size = 93).sum()/93
        results.append(sample)
    return pd.Series(results)


def calc_pval():
    """
    calc_pval returns a list where:
        - the first element is the p-value for
        hypothesis test (using 100,000 simulations).
        - the second element is Reject if you reject
        the null hypothesis and Fail to reject if you
        fail to reject the null hypothesis.

    :Example:
    >>> out = calc_pval()
    >>> len(out)
    2
    >>> 0 <= out[0] <= 1
    True
    >>> out[1] in ['Reject', 'Fail to reject']
    True
    """
    return [0.0001,'Reject']


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------

# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['data_load', 'pass_fail', 'av_score',
            'highest_score_name', 'idx_dup'],
    'q02': ['trick_me', 'reason_dup', 'trick_bool', 'reason_bool'],
    'q03': ['change', 'correct_replacement'],
    'q04': ['population_stats', 'most_common'],
    'q05': ['null_hypoth', 'simulate_null', 'estimate_p_val'],
    'q06': ['super_hero_powers'],
    'q07': ['clean_heroes', 'super_hero_stats'],
    'q08': ['bhbe_col', 'observed_stat', 'simulate_bhbe_null', 'calc_pval']
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
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
