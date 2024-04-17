import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn import datasets, linear_model
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV data using pandas, specifying delimiter as ';'
df_day_0 = pd.read_csv("round_3\\prices_round_3_day_0.csv", delimiter=";")
df_day_1 = pd.read_csv("round_3\\prices_round_3_day_1.csv", delimiter=";")
df_day_2 = pd.read_csv("round_3\\prices_round_3_day_2.csv", delimiter=";")

df_day_combined = pd.concat([df_day_0,df_day_1,df_day_2])

day_comb_cc = df_day_combined[df_day_combined['product'] == 'CHOCOLATE'].reset_index().add_suffix('_cc')
day_comb_sb = df_day_combined[df_day_combined['product'] == 'STRAWBERRIES'].reset_index().add_suffix('_sb')
day_comb_rs = df_day_combined[df_day_combined['product'] == 'ROSES'].reset_index().add_suffix('_rs')
day_comb_gb = df_day_combined[df_day_combined['product'] == 'GIFT_BASKET'].reset_index().add_suffix('_gb')


a = pd.concat([day_comb_cc['mid_price_cc'],day_comb_sb['mid_price_sb'],day_comb_rs['mid_price_rs'],day_comb_gb['mid_price_gb']],axis=1)
a['tot_price'] = 4*a['mid_price_cc'] + 6*a['mid_price_sb'] + a['mid_price_rs']
a['diff'] = a['mid_price_gb'] - 4*a['mid_price_cc'] - 6*a['mid_price_sb'] - a['mid_price_rs']
a 
print(f"mean:{np.mean(a['diff'])};median{np.median(a['diff'])};std{np.std(a['diff'])}")

sns.distplot(a['diff'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()



day_comb_cc['next_price'] = day_comb_cc['mid_price_cc'].shift(-1)
day_comb_sb['next_price'] = day_comb_sb['mid_price_sb'].shift(-1)
day_comb_rs['next_price'] = day_comb_rs['mid_price_rs'].shift(-1)
day_comb_gb['next_price'] = day_comb_gb['mid_price_gb'].shift(-1)

day_comb_cc['nom_change_cc'] = (day_comb_cc['next_price']-day_comb_cc['mid_price_cc'])
day_comb_sb['nom_change_sb'] = (day_comb_sb['next_price']-day_comb_sb['mid_price_sb'])
day_comb_rs['nom_change_rs'] = (day_comb_rs['next_price']-day_comb_rs['mid_price_rs'])
day_comb_gb['nom_change_gb'] = (day_comb_gb['next_price']-day_comb_gb['mid_price_gb'])

day_comb_cc['change_cc'] = (day_comb_cc['next_price']-day_comb_cc['mid_price_cc'])/day_comb_cc['mid_price_cc']
day_comb_sb['change_sb'] = (day_comb_sb['next_price']-day_comb_sb['mid_price_sb'])/day_comb_sb['mid_price_sb']
day_comb_rs['change_rs'] = (day_comb_rs['next_price']-day_comb_rs['mid_price_rs'])/day_comb_rs['mid_price_rs']
day_comb_gb['change_gb'] = (day_comb_gb['next_price']-day_comb_gb['mid_price_gb'])/day_comb_gb['mid_price_gb']

b = pd.concat([day_comb_cc[['day_cc','timestamp_cc','mid_price_cc','nom_change_cc','change_cc']],day_comb_sb[['mid_price_sb','nom_change_sb','change_sb']],day_comb_rs[['mid_price_rs','nom_change_rs','change_rs']],day_comb_gb[['mid_price_gb','nom_change_gb','change_gb']]],axis=1)
b = b[b['timestamp_cc'] != 999900]
b

b.corr()

b[[
'nom_change_cc'
,'nom_change_sb'
,'nom_change_rs'
,'nom_change_gb'
,'change_cc'
,'change_sb'
,'change_rs'
,'change_gb'
,'mid_price_cc'
,'mid_price_sb'
,'mid_price_rs'
,'mid_price_gb'
]].describe()




corr = b[['change_cc','change_sb','change_rs','change_gb']].corr(method = 'pearson')
corr

