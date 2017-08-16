import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data

"""%matplotlib inline"""
plt.rcParams['figure.figsize'] = 8,6

# Set stock and data source
stock = 'GOOG'
source = 'yahoo'

# Set date range (Google went public August 19, 2004)
start = datetime.datetime(2004, 8, 19)
end = datetime.datetime(2016, 12, 31)

# Collect Google stock data
goog_df = data.DataReader(stock, source, start, end)
goog_df.head()