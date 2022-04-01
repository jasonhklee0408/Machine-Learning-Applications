
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project,
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type.
    For example the lab assignments all have names of the form labXX where XX
    is a zero-padded two digit number. See the doctests for more details.

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    points = {'lab': [], 'project': [], 'midterm': [],
                'final': [], 'disc': [], 'checkpoint': []}

    for each in grades.columns:
        if 'lab' in each and '-' not in each:
            points['lab'].append(each)
        elif 'project' in each:
            if '-' not in each and '_' not in each:
                points['project'].append(each)
            elif 'checkpoint' in each and '-' not in each:
                points['checkpoint'].append(each)
        elif each == 'Midterm':
            points['midterm'].append(each)
        elif each == 'Final':
            points['final'].append(each)
        elif 'discussion' in each and '-' not in each:
            points['disc'].append(each)
    return points


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus.
    The output Series should contain values between 0 and 1.

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    projects = get_assignment_names(grades)['project']
    free_response = [each for each in grades.columns if
                        'free_response' in each and '-' not in each]

    grades = grades.fillna(0)
    raw = []
    for each in projects:
        if not any(each in piece for piece in free_response):
            raw.append((grades[each] / grades[each + ' - Max Points']))
        else:
            for every in free_response:
                if each in every:
                    raw.append(((grades[each] + grades[every]) /
                            (grades[each + ' - Max Points'] +
                                grades[every + ' - Max Points'])))
    # pd.DataFrame(raw).T.mean(axis=1)
    return pd.DataFrame(raw).T.mean(axis=1)
# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe
    grades and a Series indexed by lab assignment that
    contains the number of submissions that were turned
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """
    please_fix = []
    errors = []
    lateness = [each for each in grades.columns if 'lab' in each and
                    'Lateness' in each]
    for each in lateness:
        "number of hours late for each submission"
        please_fix.append(grades[each].str.split(':'))

    for lab in please_fix:
        for time in range(len(lab)):
            if (int(lab[time][0]) > 7):
                lab = lab.drop([time])
            elif int(lab[time][0]) == 0:
                if int(lab[time][1]) == 0 and int(lab[time][2]) == 0:
                    lab = lab.drop([time])
        errors.append(len(lab))
    results = pd.Series(errors, index = get_assignment_names(grades)['lab'])
    return results


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """
    lateness = []
    for each in col:
        time = each.split(':')
        check = float(time[0])
        if check < 7:
            lateness.append(1.0)
        elif check > 7:
            if 0 <= (check / 24) <= 7:
                lateness.append(0.9)
            elif 7 < (check / 24) <= 14:
                lateness.append(0.7)
            else:
                lateness.append(0.4)
    return pd.Series(lateness)


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment,
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    labs = grades.filter(like='lab', axis=1)
    for lab in get_assignment_names(labs)['lab']:
        total = labs[list(filter(lambda x: lab in x and 'Max' in x, labs.columns))[0]]
        late = labs[list(filter(lambda x: lab in x and 'Late' in x, labs.columns))[0]]
        new = (labs[lab] / total) * lateness_penalty(late)
        labs.loc[:,lab] = new
    labs = labs.loc[:,~labs.columns.str.contains('-')]
    return labs


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series).

    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    processed.values.sort()
    return  processed.drop(processed.columns[0],
                axis=1).mean(axis=1)


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    grades = grades.fillna(0)
    lab = lab_total(process_labs(grades)) * 0.2
    project = projects_total(grades) * 0.3
    def totals(grades, category, percentage):
        raw = grades[list(filter(lambda x: category in x and '-' not in x,
                        grades.columns))].sum(axis=1)
        total = grades[list(filter(lambda x: category in x and 'Max' in x,
                        grades.columns))].sum(axis=1)
        return (raw / total) * percentage
    assignments = ['checkpoint', 'discussion', 'Midterm', 'Final']
    percentages = [0.025, 0.025, 0.15, 0.3]
    scores = list(map(lambda x, y : totals(grades, x, y), assignments, percentages))
    scores.extend([lab, project])
    return pd.DataFrame(scores).T.sum(axis=1)


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    def letter(percent):
        if percent >= 0.90:
            return 'A'
        elif 0.90 > percent >= 0.80:
            return 'B'
        elif 0.80 > percent >= 0.70:
            return 'C'
        elif 0.70 > percent >= 0.60:
            return 'D'
        elif percent < 0.60:
            return 'F'
    return total.apply(letter)


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    scores = final_grades(total_points(grades)).value_counts()
    out = scores / scores.sum()
    return out
# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of seniors
    was worse than the class under null hypothesis conditions
    (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """
    grades['total'] = total_points(grades)
    overall = grades['total'].mean()
    seniors = grades[grades['Level'] == 'SR']['total']
    sr_mean = seniors.mean()
    sim = pd.Series(np.mean(grades.sample(len(seniors), replace=False)['total']) for i in range(N))
    return (sim <= sr_mean).mean()


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades,
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    def noise_projects_total(grades):
        projects = get_assignment_names(grades)['project']
        free_response = [each for each in grades.columns if
                            'free_response' in each and '-' not in each]

        grades = grades.fillna(0)
        raw = []
        for each in projects:
            if not any(each in piece for piece in free_response):
                to_add = (grades[each] / grades[each + ' - Max Points'])
                raw.append(np.clip(to_add + np.random.normal(0,0.02, size=(len(to_add))), 0,100.0))
            else:
                for every in free_response:
                    if each in every:
                        add = ((grades[each] + grades[every]) /
                                (grades[each + ' - Max Points'] +
                                    grades[every + ' - Max Points']))
                        raw.append(np.clip(add + np.random.normal(0,0.02, size=(len(add))), 0,100.0))
        # pd.DataFrame(raw).T.mean(axis=1)
        return pd.DataFrame(raw).T.mean(axis=1)

    def noise_process_labs(grades):
        labs = grades.filter(like='lab', axis=1)
        for lab in get_assignment_names(labs)['lab']:
            total = labs[list(filter(lambda x: lab in x and 'Max' in x, labs.columns))[0]]
            late = labs[list(filter(lambda x: lab in x and 'Late' in x, labs.columns))[0]]
            labs[lab] = (labs[lab] / total) * lateness_penalty(late)
            labs[lab] = np.clip(labs[lab] + np.random.normal(0,0.02,size=(len(labs[lab]))),0,100.00)
        labs = labs.loc[:,~labs.columns.str.contains('-')]
        return labs

    def noise_total_points(grades):
        lab = lab_total(noise_process_labs(grades)) * 0.2
        project = noise_projects_total(grades) * 0.3

        def noise_totals(grades, category, percentage):
            raw = grades[list(filter(lambda x: category in x and '-' not in x,
                        grades.columns))].fillna(0)
            total = grades[list(filter(lambda x: category in x and 'Max' in x,
                        grades.columns))]
            prop = raw.div(total.values)
            return np.clip(prop + np.random.normal(0,0.02,size=(len(prop),len(prop.columns))),0,100.00).mean(axis=1) * percentage

        assignments = ['checkpoint', 'discussion', 'Midterm', 'Final']
        percentages = [0.025, 0.025, 0.15, 0.3]
        scores = list(map(lambda x, y : noise_totals(grades, x, y), assignments, percentages))
        scores.extend([lab, project])
        return pd.DataFrame(scores).T.sum(axis=1)
    return noise_total_points(grades)


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """
    # [0.006498989580234818, 77.2, [72.7, 79.44], 0.08224299065420561, [True, False]]
    # [0.00871959651648803, 63.36, [62.22, 68.89], 0.11214953271028037, [True, False]]
    # abs(total_points(grades) - total_points_with_noise(grades)).mean()
    # (abs(total_points(grades) - total_points_with_noise(grades)) < 0.01).value_counts()[True] / len(grades)
    # np.percentile([((abs(total_points(grades) - total_points_with_noise(grades))) < 0.01).value_counts()[True] / len(grades) for i in range(100)], 2.5)
    # np.percentile([((abs(total_points(grades) - total_points_with_noise(grades))) < 0.01).value_counts()[True] / len(grades) for i in range(100)], 97.5)
    # (final_grades(total_points(grades)) == final_grades(total_points_with_noise(grades))).value_counts()[False] / len(grades)
    return [0.006498989580234818, 77.2, [72.7, 79.44], 0.08224299065420561, [True, False]]
# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
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
