from sklearn.svm import SVC
from sklearn.metrics import scorer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web

Df = web.DataReader('FB', data_source='iex')
Df = Df[['Open','High','Low','Close']]

Df['high'] = Df['High'].shift(1)
Df['low'] = Df['Low'].shift(1)
Df['close'] = Df['Close'].shift(1)
Df['Signal']=1
Df['Signal'][Df['Open'].shift(-1)<Df['Open']]=-1

print(Df.head())
Df = Df.dropna()

X = Df[['Open','high','low','close']]

t = .8
split = int(t*len(Df))

reg = SVC(C=1,cache_size=200,class_weight=None,coef0=0,decision_function_shape=None,degree=3,gamma='auto',kernel='rbf',max_iter=1000,probability=False,random_state=None,shrinking=True,tol=0.001,verbose=False)
reg.fit(X[:split],y[:split])
y_predict = reg.predict(X[split:])

Df = Df.assign(P_Trend=pd.Series(np.zeros(len(X))).values)
Df['P_Trend'][split:] = y_predict
accuracy = scorer.accuracy_score(Df['Signal'][split:],Df['P_Trend'][split:])

Df = Df.assign(Ret=pd.Series(np.zeros(len(X))).values)
Df['Ret'] = np.log(Df['Open'].shift(-1)/Df['Open'])

Df = Df.assign(Ret1=pd.Series(np.zeros(len(X))).values)
Df['Ret1'] = Df['P_Trend']*Df['Ret']

Df = Df.assign(Cu_Ret1=pd.Series(np.zeros(len(X))).values)
Df['Cu_Ret1'] = np.cumsum(Df['Ret1'][split:])

Df = Df.assign(Cu_Ret=pd.Series(np.zeros(len(X))).values)
Df['Cu_Ret'] = np.cumsum(Dp['Ret'][split:])



Std = pd.expanding_std(Df['Cu_Ret1'])
Sharpe = (Df['Cu_Ret1']-Df['Cu_Ret'])/Std
Sharpe = Sharpe[split:].mean()
print('\n\nAccuracy:',accuracy)
plt.plot(Df['Cu_Ret1'],color='r',label='Strategy Returns')
plt.plot(Df['Cu_Ret'],color='g',label='Market Returns')
plt.figtext(0.14,0.7,s='Sharpe ratio: %.2f'%Sharpe)
plt.legend(log='best')
