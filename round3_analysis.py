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
df_day_combined['new_ts'] = (df_day_combined['timestamp'])+1000000*df_day_combined['day']



day_comb_cc = df_day_combined[df_day_combined['product'] == 'CHOCOLATE'].reset_index().add_suffix('_cc')
day_comb_sb = df_day_combined[df_day_combined['product'] == 'STRAWBERRIES'].reset_index().add_suffix('_sb')
day_comb_rs = df_day_combined[df_day_combined['product'] == 'ROSES'].reset_index().add_suffix('_rs')
day_comb_gb = df_day_combined[df_day_combined['product'] == 'GIFT_BASKET'].reset_index().add_suffix('_gb')


df_day_combined_long = pd.concat([day_comb_cc,day_comb_sb,day_comb_rs,day_comb_gb], axis=1)
df_day_combined_long.reset_index()
df_day_combined_long['mid_price_gbb'] = df_day_combined_long['mid_price_cc']*4 + df_day_combined_long['mid_price_sb']*6 +df_day_combined_long['mid_price_rs']*1
df_day_combined_long['mid_price_rss'] = df_day_combined_long['mid_price_gb'] - df_day_combined_long['mid_price_cc']*4 - df_day_combined_long['mid_price_sb']*6 -379.5
df_day_combined_long['mid_price_ccc'] = (df_day_combined_long['mid_price_gb'] - df_day_combined_long['mid_price_rs'] - df_day_combined_long['mid_price_sb']*6)/4 
df_day_combined_long['mid_price_sbb'] = (df_day_combined_long['mid_price_gb'] - df_day_combined_long['mid_price_rs'] - df_day_combined_long['mid_price_cc']*4 - 379.5)/6 

df_day_combined_long['norm_mid_price_gb'] = df_day_combined_long['mid_price_gb']/df_day_combined_long['mid_price_gb'].max()
df_day_combined_long['norm_mid_price_rs'] = df_day_combined_long['mid_price_rs']/df_day_combined_long['mid_price_rs'].max()
df_day_combined_long['norm_mid_price_cc'] = df_day_combined_long['mid_price_cc']/df_day_combined_long['mid_price_cc'].max()
df_day_combined_long['norm_mid_price_sb'] = df_day_combined_long['mid_price_sb']/df_day_combined_long['mid_price_sb'].max()

df_day_combined_long['norm_mid_price_rscc'] = df_day_combined_long['norm_mid_price_rs']+df_day_combined_long['norm_mid_price_cc']
df_day_combined_long['norm_mid_price_rssb'] = df_day_combined_long['norm_mid_price_rs']+df_day_combined_long['norm_mid_price_sb']
df_day_combined_long['norm_mid_price_sbcc'] = df_day_combined_long['norm_mid_price_sb']+df_day_combined_long['norm_mid_price_cc']

df_day_combined_long['new_ts'] = (df_day_combined_long['timestamp_gb'])+1000000*df_day_combined_long['day_gb']

df = df_day_combined_long[20001:30000]

df['cumsum_sb'] = df['norm_mid_price_sb'].cumsum()
df['average_sb'] = df['cumsum_sb']/(df['timestamp_sb']/100)


df['cumsum_rscc'] = df['norm_mid_price_rscc'].cumsum()
df['cumsum_rssb'] = df['norm_mid_price_rssb'].cumsum()
df['cumsum_sbcc'] = df['norm_mid_price_sbcc'].cumsum()
df['average_rscc'] = df['cumsum_rscc']/(df['timestamp_sb']/100)
df['average_rssb'] = df['cumsum_rssb']/(df['timestamp_sb']/100)
df['average_sbcc'] = df['cumsum_sbcc']/(df['timestamp_sb']/100)

df['average_sb5'] = df['norm_mid_price_sb'].rolling(5).mean()

df['average_sb200'] = df['norm_mid_price_sb'].rolling(200).mean()
df['average_rs200'] = df['norm_mid_price_rs'].rolling(200).mean()
df['average_cc200'] = df['norm_mid_price_cc'].rolling(200).mean()

# Remove NULL values
df.dropna(inplace = False, axis=0) 
# Print DataFrame


labels = ['average_sb5','norm_mid_price_sb','average_rs200','norm_mid_price_rs','average_cc200','norm_mid_price_cc']
plt.plot(df[['new_ts_cc']], df[labels])
plt.legend(labels)
plt.show()

df['diff_rs'] = df['norm_mid_price_rs'] - df['average_rs200']

strin = 'diff_cc'
print(f"mean:{np.mean(df[strin])};median{np.median(df[strin])};std{np.std(df[strin])}")

sns.distplot(df[strin], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()




df_day_combined_long['diff_gb'] = df_day_combined_long['mid_price_gbb'] - df_day_combined_long['mid_price_gb']
df_day_combined_long['diff_rs'] = df_day_combined_long['mid_price_rss'] - df_day_combined_long['mid_price_rs']
df_day_combined_long['diff_cc'] = df_day_combined_long['mid_price_ccc'] - df_day_combined_long['mid_price_cc']
df_day_combined_long['diff_sb'] = df_day_combined_long['mid_price_sbb'] - df_day_combined_long['mid_price_sb']

strin = 'diff_cc'
print(f"mean:{np.mean(df_day_combined_long[strin])};median{np.median(df_day_combined_long[strin])};std{np.std(df_day_combined_long[strin])}")

sns.distplot(df_day_combined_long[strin], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()

df_day_combined_long['tot_price'] = 4*df_day_combined_long['mid_price_cc'] + 6*df_day_combined_long['mid_price_sb'] + df_day_combined_long['mid_price_rs']
df_day_combined_long['diff'] = df_day_combined_long['mid_price_gb'] - 4*df_day_combined_long['mid_price_cc'] - 6*df_day_combined_long['mid_price_sb'] - df_day_combined_long['mid_price_rs']
a = df_day_combined_long[2000:3000]
print(f"mean:{np.mean(a['diff'])};median{np.median(a['diff'])};std{np.std(a['diff'])}")

sns.distplot(a['diff'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()

df_day_combined_long['mid_price_gb-379.5'] = df_day_combined_long['mid_price_gb'] - 379.5
df_day_combined_long['mid_price_cc-94.872'] = df_day_combined_long['mid_price_cc'] + 94.872


df = df_day_combined_long
labels = ['mid_price_ccc','mid_price_cc-94.872']
plt.plot(df[['new_ts']], df[labels])
plt.legend(labels)
plt.show()

df = df_day_combined_long
labels = ['mid_price_gbb','mid_price_gb-379.5']
plt.plot(df[['new_ts']], df[labels])
plt.legend(labels)
plt.show()




 = a['mid_price_gb'] + 379.5 - 4*a['mid_price_cc'] - 6*a['mid_price_sb'] - a['mid_price_rs']
a['pred_cc'] = (a['mid_price_gb'] - 6*a['mid_price_sb'] - a['mid_price_rs'])/4
a['diff_cc'] = a['pred_cc'] - a['mid_price_cc']

print(f"mean:{np.mean(a['diff_cc'])};median{np.median(a['diff_cc'])};std{np.std(a['diff_cc'])}")

sns.distplot(a['diff_cc'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()

a['pred_sb'] = (a['mid_price_gb'] - 4*a['mid_price_cc'] - a['mid_price_rs'])/6
a['diff_sb'] = a['pred_sb'] - a['mid_price_sb']

print(f"mean:{np.mean(a['diff_sb'])};median{np.median(a['diff_sb'])};std{np.std(a['diff_sb'])}")

sns.distplot(a['diff_cc'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()

a['pred_rs'] = (a['mid_price_gb'] - 4*a['mid_price_cc'] - 6*a['mid_price_sb'])
a['diff_rs'] = a['pred_rs'] - a['mid_price_rs']

print(f"mean:{np.mean(a['diff_rs'])};median{np.median(a['diff_rs'])};std{np.std(a['diff_rs'])}")

sns.distplot(a['diff_cc'], hist=True, kde=True, 
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

