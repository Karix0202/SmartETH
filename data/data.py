import quandl
import pandas as pd
import math

df = quandl.get('BITFINEX/ETHUSD')
df.to_csv('data.csv')