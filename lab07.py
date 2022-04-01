import os
import pandas as pd
import numpy as np
import requests
import json
import re



# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


####################
#  Regex
####################



# ---------------------------------------------------------------------
# Problem 1
# ---------------------------------------------------------------------

def match_1(string):
    """
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    #Your Code Here
    pattern = "^.{2}\[.{2}\].*"

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_2(string):
    """
    Phone numbers that start with '(858)' and
    follow the format '(xxx) xxx-xxxx' (x represents a digit)
    Notice: There is a space between (xxx) and xxx-xxxx

    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    #Your Code Here
    pattern = "^\(858\) [0-9]{3}-[0-9]{4}$"

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None




def match_3(string):
    """
    Find a pattern whose length is between 6 to 10
    and contains only word character, white space and ?.
    This string must have ? as its last character.

    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    #Your Code Here

    pattern = "^([a-zA-z]| |\?){5,9}\?$"

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    A string that begins with '$' and with another '$' within, where:
        - Characters between the two '$' can be anything except the
        letters 'a', 'b', 'c' (lower case).
        - Characters after the second '$' can only have any number
        of the letters 'a', 'b', 'c' (upper or lower case), with every
        'a' before every 'b', and every 'b' before every 'c'.
            - E.g. 'AaBbbC' works, 'ACB' doesn't.

    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False

    >>> match_4("$iiuABc")
    False
    >>> match_4("123$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    """
    #Your Code Here
    pattern = "^\$[^a-c]*\$((A|a)+(B|b)+(C|c)+)$"

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    A string that represents a valid Python file name including the extension.
    *Notice*: For simplicity, assume that the file name contains only letters, numbers and an underscore `_`.

    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """

    #Your Code Here
    pattern = "([a-zA-z]|[0-9]|_)+(\.py)$"

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    Find patterns of lowercase letters joined with an underscore.
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """

    #Your Code Here
    pattern = "^([a-z])+_([a-z])+$"

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_7(string):
    """
    Find patterns that start with and end with a _
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """

    pattern = "^_+.*_$"

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None




def match_8(string):
    """
    Apple registration numbers and Apple hardware product serial numbers
    might have the number "0" (zero), but never the letter "O".
    Serial numbers don't have the number "1" (one) or the letter "i".

    Write a line of regex expression that checks
    if the given Serial number belongs to a genuine Apple product.

    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """

    pattern = "^([^O1i])+$"

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_9(string):
    '''
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    '''


    pattern = "(CA-[0-9]{2}-(LAX|SAN)-[0-9]{4})|(NY-[0-9]{2}-[A-Z]{3}-[0-9]{4})"

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    Given an input string, cast it to lower case, remove spaces/punctuation,
    and return a list of every 3-character substring that satisfy the following:
        - The first character doesn't start with 'a' or 'A'
        - The last substring (and only the last substring) can be shorter than
        3 characters, depending on the length of the input string.

    >>> match_10('ABCdef')
    ['def']
    >>> match_10(' DEFaabc !g ')
    ['def', 'cg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10( "Ab..DEF")
    ['def']

    '''
    output = []
    lowered = string.lower().replace(" ", "")
    lowered = re.sub('a..', '', lowered)
    lowered = re.sub(r'[^\w\s]', "", lowered)
    for i in range(0, len(lowered), 3):
        substring = lowered[i:i+3]
        output.append(substring)
    return output



# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def extract_personal(s):
    """
    :Example:
    >>> fp = os.path.join('data', 'messy.test.txt')
    >>> s = open(fp, encoding='utf8').read()
    >>> emails, ssn, bitcoin, addresses = extract_personal(s)
    >>> emails[0] == 'test@test.com'
    True
    >>> ssn[0] == '423-00-9575'
    True
    >>> bitcoin[0] == '1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2'
    True
    >>> addresses[0] == '530 High Street'
    True
    """
    email = re.findall("\w+@\w+.\w+", s)
    ssn = re.findall("[0-9]{3}-[0-9]{2}-[0-9]{4}", s)
    btc = re.findall("bitcoin:(\w{34})", s)
    addy = re.findall("[0-9]+ [A-Z][a-z]+ [A-Z][a-z]+", s)
    return (email, ssn, btc, addy)

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def tfidf_data(review, reviews):
    """
    :Example:
    >>> fp = os.path.join('data', 'reviews.txt')
    >>> reviews = pd.read_csv(fp, header=None, squeeze=True)
    >>> review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
    >>> out = tfidf_data(review, reviews)
    >>> out['cnt'].sum()
    85
    >>> 'before' in out.index
    True
    """
    output = pd.Series(review.split()).value_counts().to_frame('cnt')
    output['tf'] = output['cnt']/output['cnt'].sum()
    reviews_df = reviews.to_frame('documents')
    idf = []
    for word in output.index:
        word_formatting = ' %s ' %word
        frequency = reviews_df['documents'].str.contains(word_formatting).sum()
        idf.append(np.log(len(reviews)/frequency))
    output['idf'] = idf
    output['tfidf'] = output['tf']*output['idf']
    return output


def relevant_word(out):
    """
    :Example:
    >>> fp = os.path.join('data', 'reviews.txt')
    >>> reviews = pd.read_csv(fp, header=None, squeeze=True)
    >>> review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
    >>> out = tfidf_data(review, reviews)
    >>> relevant_word(out) in out.index
    True
    """
    sort = out.sort_values(by = 'tfidf', ascending = False)
    return sort['tfidf'].index[0]


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def hashtag_list(tweet_text):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = pd.DataFrame(testdata, columns=['text'])
    >>> out = hashtag_list(test['text'])
    >>> (out.iloc[0] == ['NLP', 'NLP1', 'NLP1'])
    True
    """
    pattern = "#(\w+)"
    out = []
    for tweet in tweet_text:
        out.append(re.findall(pattern ,tweet))
    output = pd.Series(out)
    return output


def most_common_hashtag(tweet_lists):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = hashtag_list(pd.DataFrame(testdata, columns=['text'])['text'])
    >>> most_common_hashtag(test).iloc[0]
    'NLP1'
    """
    def helper_func(entry):
        if len(entry) == 0:
            return np.NaN
        elif len(entry) == 0:
            return entry[0]
        else:
            return max(set(entry), key=entry.count)

    return tweet_lists.apply(helper_func)


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------
def tags_list(tweet_text):
    pattern = "@(\w+)"
    out = []
    for tweet in tweet_text:
        out.append(re.findall(pattern ,tweet))
    output = pd.Series(out)
    return output

def links_list(tweet_text):
    pattern = "https{0,1}://[^ ]+"
    out = []
    for tweet in tweet_text:
        out.append(re.findall(pattern ,tweet))
    output = pd.Series(out)
    return output

def retweet_list(tweet_text):
    pattern = "^RT"
    out = []
    for tweet in tweet_text:
        out.append(re.findall(pattern ,tweet))
    output = pd.Series(out)
    output = output.apply(lambda x: True if len(x) > 0 else False)
    return output

def clean_text(tweet_text):
    out = []
    for entry in tweet_text:
        output = re.sub("(^RT)|(@(\w+))|(https{0,1}://[^ ]+)|(#(\w+))", " ", entry)
        output = re.sub("[^\w]", " ", output)
        output = output.strip()
        output = output.lower()
        output = re.sub(" +", " ", output)
        out.append(output)
    return out

def create_features(ira):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = pd.DataFrame(testdata, columns=['text'])
    >>> out = create_features(test)
    >>> anscols = ['text', 'num_hashtags', 'mc_hashtags', 'num_tags', 'num_links', 'is_retweet']
    >>> ansdata = [['text cleaning is cool', 3, 'NLP1', 1, 1, True]]
    >>> ans = pd.DataFrame(ansdata, columns=anscols)
    >>> (out == ans).all().all()
    True
    """
    hash_list = hashtag_list(ira['text'])
    num_hashtags = list(hash_list.apply(len))
    mc_hashtags = list(most_common_hashtag(hash_list))
    num_tags = list(tags_list(ira['text']).apply(len))
    num_links = list(links_list(ira['text']).apply(len))
    is_retweet = list(retweet_list(ira['text']))
    text = clean_text(ira['text'])
    outcome = pd.DataFrame({"text":text, "num_hashtags":num_hashtags, 'mc_hashtags':mc_hashtags, 'num_tags':num_tags, "num_links":num_links, "is_retweet":is_retweet}, index = ira.index)
    return outcome

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['match_%d' % x for x in range(1, 10)],
    'q02': ['extract_personal'],
    'q03': ['tfidf_data', 'relevant_word'],
    'q04': ['hashtag_list', 'most_common_hashtag'],
    'q05': ['create_features']
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
