import pandas as pd 
import quandl, math, datetime, time
import numpy as np
# Preprocessing is for scaling features.
# The goal is to get them in a interval of -1 to 1, helps accuracy
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get("WIKI/GOOGL")

df  = df [ ['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'] ]

#percent volatility of the stock
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#columns we care about
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] 

# creating our label
forecast_column = 'Adj. Close'
df.fillna(-99999, inplace=True)

# Ajd Price 10 days into the future
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_column].shift(-forecast_out)


#features
x = np.array(df.drop(['label'], 1))

# you always need to scale with your role data, sometimes this step is 
# optional when you're dealing with realtime data and such because it adds to processing time.
x = preprocessing.scale(x)

# Prediction data
x_lately = x[-forecast_out:]

#features
x = x[:-forecast_out:]

df.dropna(inplace=True)
# labels
y = np.array(df['label'])

# test_size = 20% of data as testing data
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
#train your model 
clf.fit(x_train, y_train)

#saving your model
with open('lenearregression.pickle', 'wb') as f:
	pickle.dump(clf, f)

#load classifier
pickle_in = open('lenearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

#test your model
accuracy = clf.score(x_test, y_test)

# In linear regression, accuracy is = error squared 
#print accuracy

#prediction
forecast_set = clf.predict(x_lately)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
