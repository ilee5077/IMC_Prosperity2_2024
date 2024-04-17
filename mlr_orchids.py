import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn import datasets, linear_model
import statsmodels.api as sm
from scipy import stats

# Read the CSV data using pandas, specifying delimiter as ';'
df_day_0 = pd.read_csv("round-2-island-data-bottle\\prices_round_2_day_-1.csv", delimiter=";")
df_day_1 = pd.read_csv("round-2-island-data-bottle\\prices_round_2_day_0.csv", delimiter=";")
df_day_2 = pd.read_csv("round-2-island-data-bottle\\prices_round_2_day_1.csv", delimiter=";")

windows = pd.concat([df_day_0[['SUNLIGHT','HUMIDITY']].shift(n) for n in range(5)], axis=1)
windows
df_day_0['SUNLIGHT_5_lag'] = pd.Series(windows.filter(like='SUNLIGHT').values.tolist(), index=df_day_0.index)
df_day_0['HUMIDITY_5_lag'] = pd.Series(windows.filter(like='HUMIDITY').values.tolist(), index=df_day_0.index)
df_day_0['HUMIDITY-1'] = df_day_0['HUMIDITY'].shift(1)
df_day_0['HUMIDITY-2'] = df_day_0['HUMIDITY'].shift(2)
df_day_0['HUMIDITY-3'] = df_day_0['HUMIDITY'].shift(3)
df_day_0['HUMIDITY-4'] = df_day_0['HUMIDITY'].shift(4)
df_day_0['HUMIDITY-5'] = df_day_0['HUMIDITY'].shift(5)
df_day_0['ORCHIDS-1'] = df_day_0['ORCHIDS'].shift(1)
df_day_0['ORCHIDS-2'] = df_day_0['ORCHIDS'].shift(2)
df_day_0['ORCHIDS-3'] = df_day_0['ORCHIDS'].shift(3)
df_day_0['ORCHIDS-4'] = df_day_0['ORCHIDS'].shift(4)
df_day_0['ORCHIDS-5'] = df_day_0['ORCHIDS'].shift(5)
df_day_0 = df_day_0[5:]

df_day_0['ORCHIDS+1'] = df_day_0['ORCHIDS'].shift(-1)
df_day_0['ORCHIDS+2'] = df_day_0['ORCHIDS'].shift(-2)
df_day_0['ORCHIDS+3'] = df_day_0['ORCHIDS'].shift(-3)
df_day_0['ORCHIDS+5'] = df_day_0['ORCHIDS'].shift(-5)
df_day_0['ORCHIDS+10'] = df_day_0['ORCHIDS'].shift(-10)
df_day_0 = df_day_0[~df_day_0['ORCHIDS+10'].isna()]


df_day_0['increasingHUMflag'] = np.where(df_day_0['HUMIDITY']>df_day_0['prevHUMIDITY'],1,0)


df_day_0 = df_day_0.reset_index()

df_day_0['SUNGLIGHT_G_5'] = 0
df_day_0['HUMIDITY_G_5'] = 0
for i in range(len(df_day_0)):
    df_day_0['SUNGLIGHT_G_5'][i],intercept = np.polyfit([5,4,3,2,1], df_day_0['SUNLIGHT_5_lag'][i], 1)
    df_day_0['HUMIDITY_G_5'][i],intercept = np.polyfit([5,4,3,2,1], df_day_0['HUMIDITY_5_lag'][i], 1)


windows = pd.concat([df_day_1[['SUNLIGHT','HUMIDITY']].shift(n) for n in range(5)], axis=1)
windows
df_day_1['SUNLIGHT_5_lag'] = pd.Series(windows.filter(like='SUNLIGHT').values.tolist(), index=df_day_1.index)
df_day_1['HUMIDITY_5_lag'] = pd.Series(windows.filter(like='HUMIDITY').values.tolist(), index=df_day_1.index)
df_day_1['ORCHIDS-1'] = df_day_1['ORCHIDS'].shift(1)
df_day_1['ORCHIDS-2'] = df_day_1['ORCHIDS'].shift(2)
df_day_1['ORCHIDS-3'] = df_day_1['ORCHIDS'].shift(3)
df_day_1['ORCHIDS-4'] = df_day_1['ORCHIDS'].shift(4)
df_day_1['ORCHIDS-5'] = df_day_1['ORCHIDS'].shift(5)
df_day_1['HUMIDITY-1'] = df_day_1['HUMIDITY'].shift(1)
df_day_1['HUMIDITY-2'] = df_day_1['HUMIDITY'].shift(2)
df_day_1['HUMIDITY-3'] = df_day_1['HUMIDITY'].shift(3)
df_day_1['HUMIDITY-4'] = df_day_1['HUMIDITY'].shift(4)
df_day_1['HUMIDITY-5'] = df_day_1['HUMIDITY'].shift(5)
df_day_1 = df_day_1[5:]

df_day_1['ORCHIDS+1'] = df_day_1['ORCHIDS'].shift(-1)
df_day_1['ORCHIDS+2'] = df_day_1['ORCHIDS'].shift(-2)
df_day_1['ORCHIDS+3'] = df_day_1['ORCHIDS'].shift(-3)
df_day_1['ORCHIDS+5'] = df_day_1['ORCHIDS'].shift(-5)
df_day_1['ORCHIDS+10'] = df_day_1['ORCHIDS'].shift(-10)
df_day_1 = df_day_1[~df_day_1['ORCHIDS+10'].isna()]

df_day_1['increasingHUMflag'] = np.where(df_day_1['HUMIDITY']>df_day_1['prevHUMIDITY'],1,0)
df_day_1 = df_day_1.reset_index()

df_day_1['SUNGLIGHT_G_5'] = 0
df_day_1['HUMIDITY_G_5'] = 0
for i in range(len(df_day_1)):
    df_day_1['SUNGLIGHT_G_5'][i],intercept = np.polyfit([5,4,3,2,1], df_day_1['SUNLIGHT_5_lag'][i], 1)
    df_day_1['HUMIDITY_G_5'][i],intercept = np.polyfit([5,4,3,2,1], df_day_1['HUMIDITY_5_lag'][i], 1)



windows = pd.concat([df_day_2[['SUNLIGHT','HUMIDITY']].shift(n) for n in range(5)], axis=1)
windows
df_day_2['SUNLIGHT_5_lag'] = pd.Series(windows.filter(like='SUNLIGHT').values.tolist(), index=df_day_2.index)
df_day_2['HUMIDITY_5_lag'] = pd.Series(windows.filter(like='HUMIDITY').values.tolist(), index=df_day_2.index)

df_day_2['HUMIDITY-1'] = df_day_2['HUMIDITY'].shift(1)
df_day_2['HUMIDITY-2'] = df_day_2['HUMIDITY'].shift(2)
df_day_2['HUMIDITY-3'] = df_day_2['HUMIDITY'].shift(3)
df_day_2['HUMIDITY-4'] = df_day_2['HUMIDITY'].shift(4)
df_day_2['HUMIDITY-5'] = df_day_2['HUMIDITY'].shift(5)
df_day_2['ORCHIDS-1'] = df_day_2['ORCHIDS'].shift(1)
df_day_2['ORCHIDS-2'] = df_day_2['ORCHIDS'].shift(2)
df_day_2['ORCHIDS-3'] = df_day_2['ORCHIDS'].shift(3)
df_day_2['ORCHIDS-4'] = df_day_2['ORCHIDS'].shift(4)
df_day_2['ORCHIDS-5'] = df_day_2['ORCHIDS'].shift(5)
df_day_2 = df_day_2[5:]

df_day_2['ORCHIDS+1'] = df_day_2['ORCHIDS'].shift(-1)
df_day_2['ORCHIDS+2'] = df_day_2['ORCHIDS'].shift(-2)
df_day_2['ORCHIDS+3'] = df_day_2['ORCHIDS'].shift(-3)
df_day_2['ORCHIDS+5'] = df_day_2['ORCHIDS'].shift(-5)
df_day_2['ORCHIDS+10'] = df_day_2['ORCHIDS'].shift(-10)
df_day_2 = df_day_2[~df_day_2['ORCHIDS+10'].isna()]

df_day_2['increasingHUMflag'] = np.where(df_day_2['HUMIDITY']>df_day_2['prevHUMIDITY'],1,0)
df_day_2 = df_day_2.reset_index()

df_day_2['SUNGLIGHT_G_5'] = 0
df_day_2['HUMIDITY_G_5'] = 0
for i in range(len(df_day_2)):
    df_day_2['SUNGLIGHT_G_5'][i],intercept = np.polyfit([5,4,3,2,1], df_day_2['SUNLIGHT_5_lag'][i], 1)
    df_day_2['HUMIDITY_G_5'][i],intercept = np.polyfit([5,4,3,2,1], df_day_2['HUMIDITY_5_lag'][i], 1)




df_combined = pd.concat([df_day_0, df_day_1, df_day_2])
df_combined

df_combined["diffSUNLIGHT>2500"] = np.where(df_combined["SUNLIGHT"] > 2500,df_combined["SUNLIGHT"]-2500,0)
df_combined["humdiff"] = np.where((df_combined["HUMIDITY"]>80), df_combined["HUMIDITY"]-80, np.where((df_combined["HUMIDITY"]<60), 60-df_combined["HUMIDITY"],0))

df_combined["flagSUNLIGHT>2500"] = np.where(df_combined["SUNLIGHT"] > 2500,1,0)
df_combined["humflag"] = np.where((df_combined["HUMIDITY"]>80), 1, np.where((df_combined["HUMIDITY"]<60), 1,0))

df_combined['buy_profit'] = -(df_combined['TRANSPORT_FEES']+df_combined['IMPORT_TARIFF'])
df_combined['sell_profit'] = -df_combined['TRANSPORT_FEES']-df_combined['EXPORT_TARIFF']

# Split the data into independent variables (X) and the dependent variable (Y)
X = df_combined[['ORCHIDS','SUNLIGHT', 'HUMIDITY', 'diffSUNLIGHT>2500', 'humdiff', 'flagSUNLIGHT>2500', 'humflag']]
X = df_combined[['ORCHIDS','ORCHIDS-1','ORCHIDS-2','ORCHIDS-3','ORCHIDS-4','ORCHIDS-5','SUNLIGHT', 'HUMIDITY','SUNGLIGHT_G_5','HUMIDITY_G_5', 'diffSUNLIGHT>2500', 'humdiff','increasingHUMflag']]
X = df_combined[['ORCHIDS','SUNLIGHT', 'HUMIDITY','SUNGLIGHT_G_5','HUMIDITY_G_5', 'diffSUNLIGHT>2500', 'humdiff','increasingHUMflag']]

X = df_combined[['ORCHIDS','SUNLIGHT', 'HUMIDITY', 'diffSUNLIGHT>2500', 'humdiff']]
X = df_combined[['ORCHIDS', 'HUMIDITY', 'HUMIDITY-1', 'HUMIDITY-2', 'diffSUNLIGHT>2500']]
#X = df_combined[['ORCHIDS', 'HUMIDITY', 'diffSUNLIGHT>2500', 'humdiff']]
Y = df_combined['ORCHIDS+1']


df_combined

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred, squared=False))

# Print the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
X[:10]
Y[:10]

df_combined.to_csv('round-2-island-data-bottle\\prices_round_2_combined_mlr.csv')


# SIGNIFICANCE TESTING
X = df[df.columns.difference(['nextORCHIDS','DAY','IMPORT_TARIFF','EXPORT_TARIFF','timestamp'])]
X = df_combined[['HUMIDITY', 'SUNLIGHT', 'diffSUNLIGHT>2500', 'humdiff','SUNGLIGHT_G_5','HUMIDITY_G_5','increasingHUMflag']]
X = df_combined[['ORCHIDS','SUNLIGHT', 'HUMIDITY', 'diffSUNLIGHT>2500', 'humdiff','TRANSPORT_FEES','EXPORT_TARIFF']]
X = df_combined[['ORCHIDS', 'HUMIDITY', 'HUMIDITY-1', 'HUMIDITY-2', 'diffSUNLIGHT>2500']]


#X = df_combined[['ORCHIDS', 'HUMIDITY', 'diffSUNLIGHT>2500', 'humdiff']]
Y = df_combined['ORCHIDS+1']
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())

from sklearn.linear_model import RidgeCV

model = RidgeCV()

model.fit(X_train, Y_train)

print(f"model score on training data: {model.score(X_train, Y_train)}")
print(f"model score on testing data: {model.score(X_test, Y_test)}")



