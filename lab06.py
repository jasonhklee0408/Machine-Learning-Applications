import os
import pandas as pd
import numpy as np
import requests
import bs4
import json


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.

    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!

    >>> os.path.exists('lab06_1.html')
    True
    """

    # Don't change this function body!
    # No python required; create the HTML file.

    return


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def extract_book_links(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'products.html')
    >>> out = extract_book_links(open(fp, encoding='utf-8').read())
    >>> url = 'scarlet-the-lunar-chronicles-2_218/index.html'
    >>> out[1] == url
    True
    """
    results = []
    soup = bs4.BeautifulSoup(text)
    products = soup.find_all('article', attrs = {'class': 'product_pod'})
    for product in products:
        if ((len(product.find_all('p', attrs = {'class':'star-rating Four'})) == 1) | (len(product.find_all('p', attrs = {'class':'star-rating Five'})) == 1)):
            if float(product.find('p', attrs = {'class':'price_color'}).text.strip('Â£')) < 50:
                url = product.find('a', href=True)['href']
                results.append(url)
    return results


def get_product_info(text, categories):
    """
    :Example:
    >>> fp = os.path.join('data', 'Frankenstein.html')
    >>> out = get_product_info(open(fp, encoding='utf-8').read(), ['Default'])
    >>> isinstance(out, dict)
    True
    >>> 'Category' in out.keys()
    True
    >>> out['Rating']
    'Two'
    """
    output = {}
    soup = bs4.BeautifulSoup(text)
    genre = soup.find_all('ul', attrs = {'class': 'breadcrumb'})[0].find_all('li')[2].text.strip()
    if genre in categories:
        product_page = soup.find('article', attrs = {'class': 'product_page'})
        output['Availability'] = product_page.find_all('td')[5].text
        output['Category'] = genre
        output['Description'] = product_page.find_all('p')[3].text
        output['Number of reviews'] = product_page.find_all('td')[6].text
        output['Price (excl. tax)'] = product_page.find_all('td')[2].text
        output['Price (incl. tax)'] = product_page.find_all('td')[3].text
        output['Product Type'] = product_page.find_all('td')[1].text
        output['Rating'] = product_page.find_all('p')[2].attrs['class'][1]
        output['Tax'] = product_page.find_all('td')[4].text
        output['Title'] = product_page.find('h1').text
        output['UPC'] =  product_page.find_all('td')[0].text
        return output
    else:
        return None



def scrape_books(k, categories):
    """
    :param k: number of book-listing pages to scrape.
    :returns: a dataframe of information on (certain) books
    on the k pages (as described in the question).

    :Example:
    >>> out = scrape_books(1, ['Mystery'])
    >>> out.shape
    (1, 11)
    >>> out['Rating'][0] == 'Four'
    True
    >>> out['Title'][0] == 'Sharp Objects'
    True
    """
    output = pd.DataFrame()
    for i in range(1, k+1):
        website = 'http://books.toscrape.com/catalogue/page-' +str(i) + '.html'
        response = requests.get(website).text
        filtered = extract_book_links(response)
        for link in filtered:
            new_link = 'http://books.toscrape.com/catalogue/' + link
            html_convert = requests.get(new_link).text
            row = get_product_info(html_convert, categories)
            output = output.append(row, ignore_index = True)
    return output


# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def stock_history(ticker, year, month):
    """
    Given a stock code and month, return the stock price details for that month
    as a dataframe

    >>> history = stock_history('BYND', 2019, 6)
    >>> history.shape == (20, 13)
    True
    >>> history.label.iloc[-1]
    'June 03, 19'
    """
    stock_endpoint = 'https://financialmodelingprep.com/api/v3/historical-price-full/{}?apikey=6d2a22dbc3429e6877626e413905820a'
    response = requests.get(stock_endpoint.format(ticker)).json()
    output = pd.DataFrame(response['historical'])
    output['date'] = pd.to_datetime(output['date'])
    start_date = str(month) +'/' + str(1) + '/' + str(year)
    end_date = str(month+1) +'/' + str(1) + '/' + str(year)
    time_range = pd.date_range(start = start_date, end = end_date)
    time_range = time_range[:-1]
    return output[(output['date'] >= pd.to_datetime(time_range[0])) & (output['date'] <= pd.to_datetime(time_range[-1]))]


def stock_stats(history):
    """
    Given a stock's trade history, return the percent change and transactions
    in billion dollars.

    >>> history = stock_history('BYND', 2019, 6)
    >>> stats = stock_stats(history)
    >>> len(stats[0]), len(stats[1])
    (7, 6)
    >>> float(stats[0][1:-1]) > 30
    True
    >>> float(stats[1][:-1]) > 1
    True
    """
    history = history.sort_values(by = ['date'], ascending = False)
    percent_change = (((history['close'].iloc[0] - history['open'].iloc[-1])/history['open'].iloc[-1])*100).round(2)
    if percent_change > 0:
        percent_change = '+' + str(percent_change) + "%"
    else:
        percent_change = str(percent_change) + "%"

    average = (history['open'] + history['close'])/2
    total_trans_vol_list = (average*history['volume'])/1000000000
    total_trans_vol = np.format_float_positional(total_trans_vol_list.sum(), precision = 2)
    total_trans_vol = str(total_trans_vol) +"B"
    return tuple([percent_change, total_trans_vol])


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def get_comments(storyid):
    """
    Returns a dataframe of all the comments below a news story
    >>> out = get_comments(18344932)
    >>> out.shape
    (18, 5)
    >>> out.loc[5, 'by']
    'RobAtticus'
    >>> out.loc[5, 'time'].day
    31
    """
    news_endpoint = "https://hacker-news.firebaseio.com/v0/item/{}.json"
    response = requests.get(news_endpoint.format(storyid)).json()
    kids_reversed = response['kids'][::-1]
    comments_list = []
    output = pd.DataFrame()

    def dfs(kids_reversed, comments_list):
        news_endpoint = "https://hacker-news.firebaseio.com/v0/item/{}.json"
        while len(kids_reversed) != 0:
            response = requests.get(news_endpoint.format(kids_reversed.pop())).json()
            comments_list.append(response)
            if 'kids' in response.keys():
                kids_response_reversed = response['kids'][::-1]
                for id in kids_response_reversed:
                    kids_reversed.append(id)
            dfs(kids_reversed, comments_list)
        return

    dfs(kids_reversed, comments_list)

    for comment in comments_list:
        output = output.append(comment, ignore_index = True)
    output = output[output['dead'].isnull()].reset_index(drop = True)
    output = output[['id', 'by','parent','text','time']]
    output['time'] = output['time'].astype(int)
    output['time'] = pd.to_datetime(output['time'], unit = 's')
    output['id'] = output['id'].astype(int)
    output['parent'] = output['parent'].astype(int)
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
    'q01': ['question1'],
    'q02': ['extract_book_links', 'get_product_info', 'scrape_books'],
    'q03': ['stock_history', 'stock_stats'],
    'q04': ['get_comments']
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
