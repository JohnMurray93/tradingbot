# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
import seaborn

# Fetch the Data
from iexfinance import get_historical_data 
from datetime import datetime


start = datetime(2017, 1, 1) # starting date: year-month-date
end = datetime(2018, 1, 1) # ending date: year-month-date

Df = get_historical_data('SPY', start=start, end=end, output_format='pandas')          
Df= Df.dropna()
Df = Df.rename (columns={'open':'Open', 'high':'High','low':'Low', 'close':'Close'})

Df.Close.plot(figsize=(10,5))
plt.ylabel("S&P500 Price")
plt.show()

y = np.where(Df['Close'].shift(-1) > Df['Close'],1,-1)

Df['Open-Close'] = Df.Open - Df.Close
Df['High-Low'] = Df.High - Df.Low
X=Df[['Open-Close','High-Low']]
X.head()

split_percentage = 0.8
split = int(split_percentage*len(Df))

# Train data set
X_train = X[:split]
y_train = y[:split] 

# Test data set
X_test = X[split:]
y_test = y[split:]

cls = SVC().fit(X_train, y_train)

accuracy_train = accuracy_score(y_train, cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))

print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))

Df['Predicted_Signal'] = cls.predict(X)
# Calculate log returns
Df['Return'] = np.log(Df.Close.shift(-1) / Df.Close)*100
Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal
Df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))
plt.ylabel("Strategy Returns (%)")
plt.show()