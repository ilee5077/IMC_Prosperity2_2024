import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn import datasets, linear_model
import statsmodels.api as sm
from scipy import stats
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

#whenever then sunlight exposure is less than 7 hours a day, production will decrease with 4% for every 10 mins
#ideal humidity for orchid to grow is between 60 and 80, outside the limit production will fall 2% for every 5% point of humidity change

# Read the CSV data using pandas, specifying delimiter as ';'
df_day_0 = pd.read_csv("round-2-island-data-bottle\\prices_round_2_day_-1.csv", delimiter=";")
df_day_1 = pd.read_csv("round-2-island-data-bottle\\prices_round_2_day_0.csv", delimiter=";")
df_day_2 = pd.read_csv("round-2-island-data-bottle\\prices_round_2_day_1.csv", delimiter=";")

df_day_0['good_sl_flag'] = np.where(df_day_0['SUNLIGHT'] >= 2500, 1, 0)
df_day_0['tot_good_sl'] = df_day_0['good_sl_flag'].cumsum()
df_day_1['good_sl_flag'] = np.where(df_day_1['SUNLIGHT'] >= 2500, 1, 0)
df_day_1['tot_good_sl'] = df_day_1['good_sl_flag'].cumsum()
df_day_2['good_sl_flag'] = np.where(df_day_2['SUNLIGHT'] >= 2500, 1, 0)
df_day_2['tot_good_sl'] = df_day_2['good_sl_flag'].cumsum()

df_day_0['ORCHIDS+5'] =         df_day_0['ORCHIDS'].shift(-5)
df_day_0['ORCHIDS+1'] =         df_day_0['ORCHIDS'].shift(-1)
df_day_0['SUNLIGHT+1'] =        df_day_0['SUNLIGHT'].shift(-1)
df_day_0['HUMIDITY+1'] =        df_day_0['HUMIDITY'].shift(-1)
df_day_0['TRANSPORT_FEES+1'] =  df_day_0['TRANSPORT_FEES'].shift(-1)
df_day_0['EXPORT_TARIFF+1'] =   df_day_0['EXPORT_TARIFF'].shift(-1)
df_day_0['IMPORT_TARIFF+1'] =   df_day_0['IMPORT_TARIFF'].shift(-1)

df_day_1['ORCHIDS+5'] =         df_day_1['ORCHIDS'].shift(-5)
df_day_1['ORCHIDS+1'] =         df_day_1['ORCHIDS'].shift(-1)
df_day_1['SUNLIGHT+1'] =        df_day_1['SUNLIGHT'].shift(-1)
df_day_1['HUMIDITY+1'] =        df_day_1['HUMIDITY'].shift(-1)
df_day_1['TRANSPORT_FEES+1'] =  df_day_1['TRANSPORT_FEES'].shift(-1)
df_day_1['EXPORT_TARIFF+1'] =   df_day_1['EXPORT_TARIFF'].shift(-1)
df_day_1['IMPORT_TARIFF+1'] =   df_day_1['IMPORT_TARIFF'].shift(-1)

df_day_2['ORCHIDS+5'] =         df_day_2['ORCHIDS'].shift(-5)
df_day_2['ORCHIDS+1'] =         df_day_2['ORCHIDS'].shift(-1)
df_day_2['SUNLIGHT+1'] =        df_day_2['SUNLIGHT'].shift(-1)
df_day_2['HUMIDITY+1'] =        df_day_2['HUMIDITY'].shift(-1)
df_day_2['TRANSPORT_FEES+1'] =  df_day_2['TRANSPORT_FEES'].shift(-1)
df_day_2['EXPORT_TARIFF+1'] =   df_day_2['EXPORT_TARIFF'].shift(-1)
df_day_2['IMPORT_TARIFF+1'] =   df_day_2['IMPORT_TARIFF'].shift(-1)

df_day_combined = pd.concat([df_day_0.dropna(inplace = False, axis=0),df_day_1.dropna(inplace = False, axis=0),df_day_2.dropna(inplace = False, axis=0)])



#df_day_combined.iloc[::5, :]
df_day_combined = df_day_combined.iloc[::5, :]

df_day_combined['or_nom_change'] = df_day_combined['ORCHIDS+5'] - df_day_combined['ORCHIDS']
df_day_combined['or_perc_change'] = df_day_combined['or_nom_change']/df_day_combined['ORCHIDS']
df_day_combined['sl_nom_change'] = df_day_combined['SUNLIGHT+1'] - df_day_combined['SUNLIGHT']
df_day_combined['sl_perc_change'] = df_day_combined['sl_nom_change']/df_day_combined['SUNLIGHT']
df_day_combined['hm_nom_change'] = df_day_combined['HUMIDITY+1'] - df_day_combined['HUMIDITY']
df_day_combined['hm_perc_change'] = df_day_combined['hm_nom_change']/df_day_combined['HUMIDITY']
df_day_combined['tf_nom_change'] = df_day_combined['TRANSPORT_FEES+1'] - df_day_combined['TRANSPORT_FEES']
df_day_combined['tf_perc_change'] = df_day_combined['tf_nom_change']/df_day_combined['TRANSPORT_FEES']
df_day_combined['et_nom_change'] = df_day_combined['EXPORT_TARIFF+1'] - df_day_combined['EXPORT_TARIFF']
df_day_combined['et_perc_change'] = df_day_combined['et_nom_change']/df_day_combined['EXPORT_TARIFF']
df_day_combined['it_nom_change'] = df_day_combined['IMPORT_TARIFF+1'] - df_day_combined['IMPORT_TARIFF']
df_day_combined['it_perc_change'] = df_day_combined['it_nom_change']/df_day_combined['IMPORT_TARIFF']
df_day_combined['hd_out'] = np.where(df_day_combined['HUMIDITY']> 80, df_day_combined['HUMIDITY'] - 80, 
                                           np.where(df_day_combined['HUMIDITY'] < 60, 60 - df_day_combined['HUMIDITY'], 0))
df_day_combined['total_iter'] = df_day_combined['timestamp']/100 + 1
df_day_combined['sl_more_than_7'] = df_day_combined['tot_good_sl']/df_day_combined['total_iter']
df_day_combined['sl_more_than_7_flag'] = np.where(df_day_combined['tot_good_sl']/df_day_combined['total_iter'] > 7/12,1,0)
df_day_combined['sl_more_than_7_flag2'] = np.where(df_day_combined['tot_good_sl']/df_day_combined['total_iter'] > 7/24,1,0)
df_day_combined['sl_more_2556_flag'] = np.where(df_day_combined['SUNLIGHT'] >= 2556,1,0)
df_day_combined['sl_more_flag'] = np.where(df_day_combined['SUNLIGHT'] >= 2557,1,0)
df_day_combined['sl_prod_reduction'] = np.where((2557 - df_day_combined['SUNLIGHT'])/60.833 * 0.04 > 0, (2557 - df_day_combined['SUNLIGHT'])/60.833 * 0.04, 0)
df_day_combined['hm_prod_reduction'] = df_day_combined['hd_out']/5 * 0.02
df_day_combined['total_prod_reduction_add'] = 1-df_day_combined['sl_prod_reduction']-df_day_combined['hm_prod_reduction']
df_day_combined['total_prod_reduction_multi'] = (1-df_day_combined['sl_prod_reduction'])*(1-df_day_combined['hm_prod_reduction'])


for i in range(400,601):
    df_day_combined[f'sl_more_2{i}_flag'] = np.where(df_day_combined['SUNLIGHT'] >= 2000+i,1,0)

cor = df_day_combined.corr()


Y = df_day_combined['or_perc_change']
X = df_day_combined[[
    'ORCHIDS'
    ,'SUNLIGHT'
    ,'HUMIDITY'
    ,'TRANSPORT_FEES'
    ,'EXPORT_TARIFF'
    ,'IMPORT_TARIFF'
    ,'sl_nom_change'
    ,'hm_nom_change'
    ,'tf_nom_change'
    ,'et_nom_change'
    ,'it_nom_change'
    ,'sl_perc_change'
    ,'hm_perc_change'
    ,'tf_perc_change'
    ,'et_perc_change'
    ,'it_perc_change'
    ,'hd_out'
    ,'sl_more_than_7'
    ,'good_sl_flag'
    ,'tot_good_sl'
    ,'sl_more_than_7_flag'
    ,'sl_more_than_7_flag2'
    ,'sl_more_2556_flag'
    ,'sl_more_flag'
    ,'sl_prod_reduction'
    ,'hm_prod_reduction'
    ,'total_prod_reduction_add'
    ,'total_prod_reduction_multi'

    ]]
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(f"coefficient of determination: {est2.rsquared}")
print(f"adjusted coefficient of determination: {est2.rsquared_adj}")
print(f"regression coefficients: {est2.params}")

Y = df_day_combined['or_perc_change']
X = df_day_combined[[
    'hm_nom_change'
    ,'hm_perc_change'
    ]]
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(f"coefficient of determination: {est2.rsquared}")
print(f"adjusted coefficient of determination: {est2.rsquared_adj}")
print(f"regression coefficients: {est2.params}")

X = df_day_combined[['hm_nom_change','hm_perc_change']]
X = sm.add_constant(X)
df_day_combined['predict'] = est2.predict(X)

df_day_combined[['or_perc_change','predict']]*100


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
print(f"coefficient of determination: {results.rsquared}")
print(f"adjusted coefficient of determination: {results.rsquared_adj}")
print(f"regression coefficients: {results.params}")
from sklearn.linear_model import RidgeCV

model = RidgeCV()

model.fit(X_train, Y_train)

print(f"model score on training data: {model.score(X_train, Y_train)}")
print(f"model score on testing data: {model.score(X_test, Y_test)}")



