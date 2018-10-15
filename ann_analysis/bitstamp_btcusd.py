import talib
import random
import numpy as py
import pandas as pd
import matplotlib.pylab as plt
from abc import ABCMeta, abstractmethod



random.seed(42)

dataset = pd.read_csv('/home/zwei/Documents/Python/ai_analysis_and_trading_bot/ann_analysis/datasets/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
dataset = dataset.dropna()
dataset = dataset[['Open','High','Low','Close']]

