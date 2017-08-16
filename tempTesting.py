# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

https://lectures.quantecon.org/py/pandas.html
"""

import matplotlib.pyplot as plt
import pandas as pd

def hello():
    """Print "Hello World" and return None"""
    print("Hello World2")

# main program starts here
hello()

print("etest")
print("new testing for commit to github")

plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()

4 + 7

df2 = pd.read_csv('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/pandas/data/test_pwt.csv')
type(df2)

print(df2)

"""print(df2[2:5])"""


""" === FED Data === """

import requests
r = requests.get('http://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv')

url = 'http://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv'
source = requests.get(url).content.decode().split("\n")
print(source[1])

data = pd.read_csv(url, index_col=0, parse_dates=True)
print(type(data))
print(data.head())

pd.set_option('precision', 1)
print(data.describe())  # Your output might differ sli

data['2006':'2012'].plot()
plt.show()
