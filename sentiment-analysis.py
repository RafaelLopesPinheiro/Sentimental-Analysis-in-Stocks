# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 11:10:02 2022

@author: rafae
"""
import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf

url_path = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG', 'NVDA']


news_tables = {} #dictionary

for ticker in tickers: 
    url = url_path + ticker
    
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    
    html = BeautifulSoup(response, features='lxml')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table
        


parsed_data = [] #list

for ticker, news_table in news_tables.items():
    
    for row in news_table.findAll('tr'):
        
        title = row.a.get_text() ## Getting </a> class in html code with text 
        date_data = row.td.text.split(' ') ## Getting the text inside </td> class in html code

        if len(date_data) == 1:  ## split date from time only
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=['Ticker','date','Time','Title'])

vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['Title'].apply(f)
df['date'] = pd.to_datetime(df.date).dt.date

plt.figure(figsize=(10,8))

mean_df = df.groupby(['Ticker', 'date']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()  # the cross-section (xs) function is to unlabel the 'compound' that came from groupby previously
# mean_df.plot(kind='bar')

df_nvda = yf.download('NVDA', start = '2022-04-07', end='2022-04-16')['Adj Close'].pct_change()
df_nvda = pd.DataFrame(df_nvda)
df_nvda['compound'] = mean_df['NVDA']
df_nvda.plot(kind='bar')


#### Conclusion ####
# Apparently no correlation between the Sentimental Analysis and the returns on NVDA stocks in such small time frame
# Could propouse testing with a large article base to a more accurate test
# Test with classification models could be done with a larger data base to make some train-test split and better evaluate the model
