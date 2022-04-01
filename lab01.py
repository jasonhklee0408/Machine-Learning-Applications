import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question # 0
# ---------------------------------------------------------------------

def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two
    adjacent elements that are consecutive integers.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two
    adjacent elements that are consecutive integers.
    :Example:
    >>> consecutive_ints([5,3,6,4,9,8])
    True
    >>> consecutive_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def median_vs_average(nums):
    """
    median takes a non-empty list of numbers,
    returning a boolean of whether the median is
    greater or equal than the average
    If the list has even length, it should return
    the mean of the two elements in the middle.
    :param nums: a non-empty list of numbers.
    :returns: bool, whether median is greater or equal than average.

    :Example:
    >>> median_vs_average([6, 5, 4, 3, 2])
    True
    >>> median_vs_average([50, 20, 15, 40])
    False
    >>> median_vs_average([1, 2, 3, 4])
    True
    """
    nums.sort()
    average = sum(nums)/len(nums)
    if len(nums) % 2 == 0:
        first_median = nums[(len(nums)//2) - 1]
        second_median = nums[len(nums)//2]
        median = (first_median + second_median) / 2
    else:
        median = nums[len(nums)//2]
    if median >= average:
        return True
    else:
        return False

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------
def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance
    as integers is also i.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two
    elements as described above.
    :Example:
    >>> same_diff_ints([5,3,1,5,9,8])
    True
    >>> same_diff_ints([1,3,5,7,9])
    False
    """
    for i in range(len(ints)):
        for j in range(i+1,len(ints)):
            difference = abs(ints[i]-ints[j])
            distance = abs(i-j)
            if difference == distance:
                return True

    return False

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def n_prefixes(s, n):
    """
    n_prefixes returns a string of n
    consecutive prefix of the input string.

    :param s: a string.
    :param n: an integer

    :returns: a string of n consecutive prefixes of s backwards.
    :Example:
    >>> n_prefixes('Data!', 3)
    'DatDaD'
    >>> n_prefixes('Marina', 4)
    'MariMarMaM'
    >>> n_prefixes('aaron', 2)
    'aaa'
    """
    output = ""
    end_index = n
    for i in range(n):
        output += s[0:end_index]
        end_index -= 1
    return output

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------
def exploded_numbers(ints, n):
    """
    exploded_numbers returns a list of strings of numbers from the
    input array each exploded by n.
    Each integer is zero padded.

    :param ints: a list of integers.
    :param n: a non-negative integer.

    :returns: a list of strings of exploded numbers.
    :Example:
    >>> exploded_numbers([3, 4], 2)
    ['1 2 3 4 5', '2 3 4 5 6']
    >>> exploded_numbers([3, 8, 15], 2)
    ['01 02 03 04 05', '06 07 08 09 10', '13 14 15 16 17']
    """
    output_list = []
    for number in ints:
        if number + 2 >= 10:
            for number in ints:
                result_stirng = ""
                starting_number = number - n
                for i in range((n*2)+1):
                    if i == n*2:
                        result_stirng += str(starting_number).zfill(2)
                    else:
                        result_stirng += str(starting_number).zfill(2) + " "
                    starting_number += 1
                output_list.append(result_stirng)
            return output_list

    for number in ints:
        result_stirng = ""
        starting_number = number - n
        for i in range((n*2)+1):
            if i == n*2:
                result_stirng += str(starting_number)
            else:
                result_stirng += str(starting_number) + " "
            starting_number += 1
        output_list.append(result_stirng)
    return output_list

# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def last_chars(fh):
    """
    last_chars takes a file object and returns a
    string consisting of the last character of the line.
    :param fh: a file object to read from.
    :returns: a string of last characters from fh
    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """
    output = ""
    lines = fh.readlines()
    for line in lines:
        output += line[-2]
    return output

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def arr_1(A):
    """
    arr_1 takes in a numpy array and
    adds to each element the square-root of
    the index of each element.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = arr_1(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    """
    square_root_list = np.arange(len(A))
    square_root_list = square_root_list**(1/2)
    output = A + square_root_list
    return output

def arr_2(A):
    """
    arr_2 takes in a numpy array of integers
    and returns a boolean array (i.e. an array of booleans)
    whose ith element is True if and only if the ith element
    of the input array is a perfect square.
    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.
    :Example:
    >>> out = arr_2(np.array([1, 2, 16, 17, 32, 49]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    """
    squared_root_list = np.sqrt(A).round()
    true_false_list = np.where(squared_root_list == np.sqrt(A), True, False)
    return true_false_list

def arr_3(A):
    """
    arr_3 takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = arr_3(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """
    difference = np.diff(A)
    division_list = A[0:len(A)-1]
    rounded_list = np.round(difference/division_list, 2)
    return rounded_list

def arr_4(A):
    """
    Create a function arr_4 that takes in A and
    returns the day on which you can buy at least
    one share from 'left-over' money. If this never
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: the day on which you can buy at least one share from 'left-over' money
    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = arr_4(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """
    budget = np.zeros(len(A))
    budget = budget + 20
    leftover = np.mod(budget, A)
    leftover_summed = np.cumsum(leftover)
    leftover_possible = np.where(A > leftover_summed, 0, 1)
    leftover_indices = np.nonzero(leftover_possible)
    if len(leftover_indices[0]) == 0:
        return -1
    else:
        return leftover_indices[0][0]

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def salary_stats(salary):
    """
    salary_stats returns a series as specified in the notebook.
    :param salary: a dataframe of NBA salaries as found in `salary.csv`
    :return: a series with index specified in the notebook.
    :Example:
    >>> salary_fp = os.path.join('data', 'salary.csv')
    >>> salary = pd.read_csv(salary_fp)
    >>> out = salary_stats(salary)
    >>> isinstance(out, pd.Series)
    True
    >>> 'total_highest' in out.index
    True
    >>> isinstance(out.loc['duplicates'], bool)
    True
    """
    try:
        num_players = salary['Player'].count()
    except:
        num_players = None
    try:
        num_teams = salary['Team'].nunique()
    except:
        num_teams = None
    try:
        total_salary = salary['Salary'].sum()
    except:
        total_salary = None
    try:
        highest_salary = salary.sort_values('Salary', ascending = False)['Player'].iloc[0]
    except:
        highest_salary = None
    try:
        boston_list = salary['Team'] == 'BOS'
        avg_bos = salary[boston_list]['Salary'].mean()
    except:
        avg_bos = None
    try:
        salary_sort = salary.sort_values('Salary')
        third_lowest = salary_sort['Player'].iloc[2] + ', ' + salary_sort['Team'].iloc[2]
    except:
        third_lowest = None
    try:
        duplicates_list = salary['Player'].str.split(expand = True)[1].nunique()
        duplicates = True
        if duplicates_list == num_players:
            duplicates = False
    except:
        duplicates = None
    try:
        highest_team = salary['Team'] == salary.sort_values('Salary',
    ascending = False)['Team'].iloc[0]
        total_highest = salary[highest_team]['Salary'].sum()
    except:
        total_highest = None
    output_series = pd.Series({'num_players': num_players,
    'num_teams':num_teams, 'total_salary':total_salary,
    'highest_salary':highest_salary, 'avg_bos':avg_bos,
    'third_lowest':third_lowest, 'duplicates':duplicates,
    'total_highest':total_highest})
    return output_series


# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def parse_malformed(fp):
    """
    Parses and loads the malformed csv data into a
    properly formatted dataframe (as described in
    the question).
    :param fh: file handle for the malformed csv-file.
    :returns: a Pandas DataFrame of the data,
    as specificed in the question statement.
    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """
    row_data = []
    with open(fp) as f:
        results = []
        for row in f:
            row = row.split(",")
            while '' in row:
                row.remove('')
            row_data.append(row)

    row_data[0] = " ".join(row_data[0]).replace('\n', '').split()

    for index in range(1, len(row_data)):
        first_name = [x for x in str(row_data[index][0]) if x.isalpha() == True]
        row_data[index][0] =  "".join(first_name)
        last_name = [x for x in str(row_data[index][1]) if x.isalpha() == True]
        row_data[index][1] = "".join(last_name)
        weight = [x for x in str(row_data[index][2]) if x.isnumeric() == True or x =="."]
        row_data[index][2] = float("".join(weight))
        height = [x for x in str(row_data[index][3]) if x.isnumeric() == True or x =="."]
        row_data[index][3] = float("".join(height))
        first_geo = [x for x in str(row_data[index][4]) if x.isnumeric() == True or x == "." or x == "-"]
        first_geo = "".join(first_geo)
        second_geo = [x for x in str(row_data[index][5]) if x.isnumeric() == True or x == "." or x == "-"]
        second_geo = "".join(second_geo)
        row_data[index][4] = ",".join((first_geo, second_geo))
        if len(row_data[index]) > 5:
            row_data[index] = row_data[index][0:5]

    f.close()
    output = pd.DataFrame(row_data[1:], columns = row_data[0])
    return output



# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------

# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q00': ['consecutive_ints'],
    'q01': ['median_vs_average'],
    'q02': ['same_diff_ints'],
    'q03': ['n_prefixes'],
    'q04': ['exploded_numbers'],
    'q05': ['last_chars'],
    'q06': ['arr_%d' % d for d in range(1, 5)],
    'q07': ['salary_stats'],
    'q08': ['parse_malformed']
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
