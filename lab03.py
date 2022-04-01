
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.

    :Example:
    >>> set(car_null_hypoth()) <= set(range(1,11))
    True
    """
    return [3,6]


def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.

    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1,11))
    True
    """
    return [2,5]


def car_test_stat():
    """
    Returns a list of valid test statistics.

    :Example:
    >>> set(car_test_stat()) <= set(range(1,5))
    True
    """
    return [2,4]


def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.

    :Example:
    >>> car_p_value() in [1,2,3,4,5]
    True
    """
    return 5


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def clean_apps(df):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> len(cleaned) == len(df)
    True
    '''
    copy = df
    copy['Reivews'] = copy['Reviews'].astype(int)
    def size_convert(x):
        if 'M' in x:
            return float(x.strip('M')) * 1000.0
        else:
            return float(x.strip('k'))

    copy['Size'] = copy['Size'].apply(size_convert)

    def installs_convert(x):
        return int(x.replace(',','').replace('+',''))

    copy['Installs'] = copy['Installs'].apply(installs_convert)

    def type_convert(x):
        if x == 'Free':
            return 1
        else:
            return 0

    copy['Type'] = copy['Type'].apply(type_convert)

    def price_convert(x):
        if '$' in x:
            return float(x.strip('$'))
        else:
            return float(x)

    copy['Price'] = copy['Price'].apply(price_convert)

    def time_convert(x):
        return int(x[-4:])

    copy['Last Updated'] = copy['Last Updated'].apply(time_convert)
    return copy


def store_info(cleaned):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> info = store_info(cleaned)
    >>> len(info)
    4
    >>> info[2] in cleaned.Category.unique()
    True
    '''
    installs_group = cleaned.groupby(['Last Updated'], as_index = False)
    hundred = installs_group['App'].count()
    hundred = list(hundred[hundred['App'] > 100]['Last Updated'])
    new_installs = list(cleaned[cleaned['Last Updated'].isin(hundred)].groupby(['Last Updated'], as_index = False)['Installs'].median().sort_values(by=['Installs'], ascending = False)['Last Updated'])[0]
    #2018
    ratings_group = cleaned.groupby(['Content Rating'])['Rating'].min().idxmax()
    #adults only 18+
    category_group = cleaned.groupby(['Category'])['Price'].mean().idxmax()
    #finance
    category_filter = cleaned[cleaned['Reviews'] >= 1000].reset_index().groupby(['Category'])['Rating'].mean().idxmin()
    #dating
    return [new_installs, ratings_group, category_group, category_filter]

# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def std_reviews_by_app_cat(cleaned):
    """
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> play = pd.read_csv(fp)
    >>> clean_play = clean_apps(play)
    >>> out = std_reviews_by_app_cat(clean_play)
    >>> set(out.columns) == set(['Category', 'Reviews'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """
    def standard_units(nums):
        convertion = (nums - np.mean(nums))/np.std(nums)
        return convertion
    cleaned['Reviews'] = cleaned.groupby(['Category'], as_index = False)['Reviews'].transform(standard_units)
    return cleaned[['Category','Reviews']]


def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0].lower() in ['medical', 'family', 'equal']
    True
    >>> out[1] in ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY',\
       'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',\
       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE',\
       'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',\
       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',\
       'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL',\
       'TOOLS', 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER',\
       'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']
    True
    """
    return ['equal', 'FAMILY']


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def read_survey(dirname):
    """
    read_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    files_list = os.listdir(dirname)
    cols = ['first name', 'last name', 'current company', 'job title', 'email', 'university']
    output = pd.DataFrame()
    for file in files_list:
        filename = os.path.join(dirname, file)
        read = pd.read_csv(filename)
        read = read.reindex(sorted(read.columns), axis = 1)
        read.columns = ['current company', 'email','first name','job title','last name','university']
        read = read.reindex(cols, axis = 1)
        output = pd.concat([output, read], sort = False, ignore_index = True)
    return output


def com_stats(df):
    """
    com_stats
    :param df: a DataFrame containing the combined survey data
    :returns: a hardcoded list of answers to the problems in the notebook
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> isinstance(out[0], int)
    True
    >>> isinstance(out[2], str)
    True
    """

    return [5,253,'Business Systems Development Analyst',369]


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def combine_surveys(dirname):
    """
    combine_surveys takes in a directory path
    (containing files favorite*.csv) and combines
    all of the survey data into one DataFrame,
    indexed by student ID (a value 0 - 1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = combine_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> combine_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    output = pd.DataFrame(columns = ['id'])
    files_list = os.listdir(dirname)
    for file in files_list:
        filename = os.path.join(dirname, file)
        read = pd.read_csv(filename)
        output = output.merge(read, on = 'id', how = 'right')
    output = output.set_index('id')
    return output


def check_credit(df):
    """
    check_credit takes in a DataFrame with the
    combined survey data and outputs a DataFrame
    of the names of students and how many extra credit
    points they would receive, indexed by their ID (a value 0-1000)

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = combine_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    """
    nan_list = df.isnull().sum(axis = 1).to_list()
    nan_list = np.divide(nan_list, len(df.columns))
    def extra_check(num):
        if num > 0.25:
            return 0
        else:
            return 5
    nan_list = list(map(extra_check, nan_list))
    for col in list(df.columns):
        null_count = df[col].isna().sum()
        if (null_count/len(df[col])) <= 0.1:
            nan_list = list(map(lambda x: x + 1, nan_list))
            break
    output = pd.DataFrame({'name':df['name'], 'points':nan_list})
    return output

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    """
    What is the most popular Procedure Type for all of the pets we have in our `pets` dataset?
​
    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = most_popular_procedure(pets, procedure_history)
    >>> isinstance(out,str)
    True
    """
    output = pd.merge(pets, procedure_history, how = 'outer')['ProcedureType'].value_counts().idxmax()
    return output


def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """
    output = pd.merge(owners, pets, on = 'OwnerID', how = 'outer',
    suffixes = ('Owner', 'Pet'))[['OwnerID','NameOwner','NamePet']].groupby(['OwnerID',
    'NameOwner'])['NamePet'].apply(list).droplevel('OwnerID')
    output = output.apply(lambda x: x[0] if len(x) == 1 else x)
    return output


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    """
    total cost per city
​
    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_city(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['City'])
    True
    """
    procedures = pd.merge(procedure_history, procedure_detail, on = ['ProcedureType','ProcedureSubCode'], how = 'left')[['PetID','Price']]
    cities = pd.merge(owners, pets, on = ['OwnerID'], how = 'left')[['PetID','City']]
    combined = pd.merge(procedures, cities, on = 'PetID').groupby('City').sum()
    return combined



# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!


GRADED_FUNCTIONS = {
    'q01': [
        'car_null_hypoth', 'car_alt_hypoth',
        'car_test_stat', 'car_p_value'
    ],
    'q02': ['clean_apps', 'store_info'],
    'q03': ['std_reviews_by_app_cat','su_and_spread'],
    'q04': ['read_survey', 'com_stats'],
    'q05': ['combine_surveys', 'check_credit'],
    'q06': ['most_popular_procedure', 'pet_name_by_owner', 'total_cost_per_city']
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
