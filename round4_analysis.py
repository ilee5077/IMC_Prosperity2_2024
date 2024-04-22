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
import math

# Read the CSV data using pandas, specifying delimiter as ';'
df_day_1 = pd.read_csv("round4\\prices_round_4_day_1.csv", delimiter=";")
df_day_2 = pd.read_csv("round4\\prices_round_4_day_2.csv", delimiter=";")
df_day_3 = pd.read_csv("round4\\prices_round_4_day_3.csv", delimiter=";")
df_day_combined = pd.concat([df_day_1,df_day_2,df_day_3])




day_1_ccn = df_day_1[df_day_1['product'] == 'COCONUT'].reset_index().add_suffix('_ccn')
day_2_ccn = df_day_2[df_day_2['product'] == 'COCONUT'].reset_index().add_suffix('_ccn')
day_3_ccn = df_day_3[df_day_3['product'] == 'COCONUT'].reset_index().add_suffix('_ccn')
day_comb_ccn = df_day_combined[df_day_combined['product'] == 'COCONUT'].reset_index().add_suffix('_ccn')

day_1_ccnc = df_day_1[df_day_1['product'] == 'COCONUT_COUPON'].reset_index().add_suffix('_ccnc')
day_2_ccnc = df_day_2[df_day_2['product'] == 'COCONUT_COUPON'].reset_index().add_suffix('_ccnc')
day_3_ccnc = df_day_3[df_day_3['product'] == 'COCONUT_COUPON'].reset_index().add_suffix('_ccnc')
day_comb_ccnc = df_day_combined[df_day_combined['product'] == 'COCONUT_COUPON'].reset_index().add_suffix('_ccnc')


df_day1_long = pd.concat([day_1_ccn,day_1_ccnc], axis=1)
df_day2_long = pd.concat([day_2_ccn,day_2_ccnc], axis=1)
df_day3_long = pd.concat([day_3_ccn,day_3_ccnc], axis=1)
df_day_comb_long = pd.concat([day_comb_ccn,day_comb_ccnc], axis=1)













'''
df_day2_long

df_day2_long['mid_price_ccn-3'] = df_day2_long['mid_price_ccn'].shift(3)
df_day2_long['mid_price_ccn-2'] = df_day2_long['mid_price_ccn'].shift(2)
df_day2_long['mid_price_ccn-1'] = df_day2_long['mid_price_ccn'].shift(1)
df_day2_long['mid_price_ccn+1'] = df_day2_long['mid_price_ccn'].shift(-1)
df_day2_long['mid_price_ccn+2'] = df_day2_long['mid_price_ccn'].shift(-2)
df_day2_long['mid_price_ccn+3'] = df_day2_long['mid_price_ccn'].shift(-3)
df_day2_long['mid_price_ccnc-3'] = df_day2_long['mid_price_ccnc'].shift(3)
df_day2_long['mid_price_ccnc-2'] = df_day2_long['mid_price_ccnc'].shift(2)
df_day2_long['mid_price_ccnc-1'] = df_day2_long['mid_price_ccnc'].shift(1)
df_day2_long['mid_price_ccnc+1'] = df_day2_long['mid_price_ccnc'].shift(-1)
df_day2_long['mid_price_ccnc+2'] = df_day2_long['mid_price_ccnc'].shift(-2)
df_day2_long['mid_price_ccnc+3'] = df_day2_long['mid_price_ccnc'].shift(-3)

df_day3_long['mid_price_ccn-3'] = df_day3_long['mid_price_ccn'].shift(3)
df_day3_long['mid_price_ccn-2'] = df_day3_long['mid_price_ccn'].shift(2)
df_day3_long['mid_price_ccn-1'] = df_day3_long['mid_price_ccn'].shift(1)
df_day3_long['mid_price_ccn+1'] = df_day3_long['mid_price_ccn'].shift(-1)
df_day3_long['mid_price_ccn+2'] = df_day3_long['mid_price_ccn'].shift(-2)
df_day3_long['mid_price_ccn+3'] = df_day3_long['mid_price_ccn'].shift(-3)
df_day3_long['mid_price_ccnc-3'] = df_day3_long['mid_price_ccnc'].shift(3)
df_day3_long['mid_price_ccnc-2'] = df_day3_long['mid_price_ccnc'].shift(2)
df_day3_long['mid_price_ccnc-1'] = df_day3_long['mid_price_ccnc'].shift(1)
df_day3_long['mid_price_ccnc+1'] = df_day3_long['mid_price_ccnc'].shift(-1)
df_day3_long['mid_price_ccnc+2'] = df_day3_long['mid_price_ccnc'].shift(-2)
df_day3_long['mid_price_ccnc+3'] = df_day3_long['mid_price_ccnc'].shift(-3)

df_day2_long = df_day2_long[['index_ccn', 'day_ccn', 'timestamp_ccn', 'mid_price_ccn', 'mid_price_ccnc', 'mid_price_ccn-3', 'mid_price_ccn-2', 'mid_price_ccn-1', 'mid_price_ccn+1', 'mid_price_ccn+2', 'mid_price_ccn+3', 'mid_price_ccnc-3', 'mid_price_ccnc-2', 'mid_price_ccnc-1', 'mid_price_ccnc+1', 'mid_price_ccnc+2', 'mid_price_ccnc+3']]
df_day3_long = df_day3_long[['index_ccn', 'day_ccn', 'timestamp_ccn', 'mid_price_ccn', 'mid_price_ccnc', 'mid_price_ccn-3', 'mid_price_ccn-2', 'mid_price_ccn-1', 'mid_price_ccn+1', 'mid_price_ccn+2', 'mid_price_ccn+3', 'mid_price_ccnc-3', 'mid_price_ccnc-2', 'mid_price_ccnc-1', 'mid_price_ccnc+1', 'mid_price_ccnc+2', 'mid_price_ccnc+3']]

df_day_combined = pd.concat([df_day2_long.dropna(inplace = False, axis=0),df_day3_long.dropna(inplace = False, axis=0)])

df_day_combined['perc_change_mid_price_ccn-3'] = ( df_day_combined['mid_price_ccn-3']  - df_day_combined['mid_price_ccn'])/df_day_combined['mid_price_ccn']
df_day_combined['perc_change_mid_price_ccn-2'] = ( df_day_combined['mid_price_ccn-2']  - df_day_combined['mid_price_ccn'])/df_day_combined['mid_price_ccn']
df_day_combined['perc_change_mid_price_ccn-1'] = ( df_day_combined['mid_price_ccn-1']  - df_day_combined['mid_price_ccn'])/df_day_combined['mid_price_ccn']
df_day_combined['perc_change_mid_price_ccn+1'] = ( df_day_combined['mid_price_ccn+1']  - df_day_combined['mid_price_ccn'])/df_day_combined['mid_price_ccn']
df_day_combined['perc_change_mid_price_ccn+2'] = ( df_day_combined['mid_price_ccn+2']  - df_day_combined['mid_price_ccn'])/df_day_combined['mid_price_ccn']
df_day_combined['perc_change_mid_price_ccn+3'] = ( df_day_combined['mid_price_ccn+3']  - df_day_combined['mid_price_ccn'])/df_day_combined['mid_price_ccn']
df_day_combined['perc_change_mid_price_ccnc-3'] = (df_day_combined['mid_price_ccnc-3'] - df_day_combined['mid_price_ccnc'])/df_day_combined['mid_price_ccnc']
df_day_combined['perc_change_mid_price_ccnc-2'] = (df_day_combined['mid_price_ccnc-2'] - df_day_combined['mid_price_ccnc'])/df_day_combined['mid_price_ccnc']
df_day_combined['perc_change_mid_price_ccnc-1'] = (df_day_combined['mid_price_ccnc-1'] - df_day_combined['mid_price_ccnc'])/df_day_combined['mid_price_ccnc']
df_day_combined['perc_change_mid_price_ccnc+1'] = (df_day_combined['mid_price_ccnc+1'] - df_day_combined['mid_price_ccnc'])/df_day_combined['mid_price_ccnc']
df_day_combined['perc_change_mid_price_ccnc+2'] = (df_day_combined['mid_price_ccnc+2'] - df_day_combined['mid_price_ccnc'])/df_day_combined['mid_price_ccnc']
df_day_combined['perc_change_mid_price_ccnc+3'] = (df_day_combined['mid_price_ccnc+3'] - df_day_combined['mid_price_ccnc'])/df_day_combined['mid_price_ccnc']

df_day_combined.corr().to_csv('correlation4.csv')
'''

df_day_comb_long['new_ts'] = (df_day_comb_long['timestamp_ccn'])+1000000*(df_day_comb_long['day_ccn']-1)
df_day_comb_long['ccn-ccnc'] = df_day_comb_long['mid_price_ccn']-df_day_comb_long['mid_price_ccnc']

df_day_comb_long['new_iter'] = df_day_comb_long['new_ts']/100

df_day_comb_long['timeval'] = df_day_comb_long['mid_price_ccnc'] - (df_day_comb_long['mid_price_ccn'] - 10000)

df_day_comb_long['mid_price_ccn'].mean()

df = df_day_comb_long
labels = ['mid_price_ccn']
plt.plot(df[['new_ts']], df[labels])
plt.legend(labels)
plt.show()

df_day_comb_long['mean_diff_ccn'] = df_day_comb_long['mid_price_ccn'] - df_day_comb_long['mid_price_ccn'].mean()
df_day_comb_long['mean_diff_ccn_squared'] = df_day_comb_long['mean_diff_ccn']**2
print(f'sd:{math.sqrt(df_day_comb_long['mean_diff_ccn_squared'].sum()/len(df_day_comb_long['mean_diff_ccn_squared']))}')
# sd:88.75266514702373
# COULD LOOK AT EACH DAYS MEAN

  """
  Calculates the Black-Scholes option price for calls or puts.

  Args:
      S (float): Spot price of the underlying asset.
      K (float): Strike price of the option.
      T (float): Time to expiration in years.
      r (float): Risk-free interest rate.
      sigma (float): Volatility of the underlying asset.
      option (str, optional): Option type ("C" for call, "P" for put). Defaults to "C".

  Returns:
      float: The Black-Scholes option price.
  """

from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option="C"):
  """
  Calculates the Black-Scholes option price for calls or puts.

  Args:
      S (float): Spot price of the underlying asset.
      K (float): Strike price of the option.
      T (float): Time to expiration in years.
      r (float): Risk-free interest rate.
      sigma (float): Volatility of the underlying asset.
      option (str, optional): Option type ("C" for call, "P" for put). Defaults to "C".

  Returns:
      float: The Black-Scholes option price.
  """

  d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  if option == "C":
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
  elif option == "P":
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
  else:
    raise ValueError("Invalid option type. Please specify 'C' for call or 'P' for put.")

  return price





S = df_day_comb_long['mid_price_ccn']
K = 10000
#r = risk-free int rate
#T = time to expiry in days
T = (250 - df_day_comb_long['day_ccn'] + 1 - (df_day_comb_long['timestamp_ccn']/1000000))/365
T=250/365
sigma = 88.75
sigma = 88.75*0.8209583440953176
sigma = 88.75*0.00217278
target_price = ['mid_price_ccnc']

# Optionally provide an initial guess for r (can affect convergence)
r = 0.0  # Initial guess for risk-free rate

df_day_comb_long['call_price'] = black_scholes(S, K, T, r, sigma, option="C")
df_day_comb_long[['call_price','mid_price_ccnc']]

curr_min = math.inf
min = -1
for i in range(215000,220000,1):
   sigma = 88.75 * (i/100000000.0)
   df_day_comb_long['call_price'] = black_scholes(S, K, T, r, sigma, option="C")
   diff = df_day_comb_long['call_price'] - df_day_comb_long['mid_price_ccnc']
   diff2 = diff**2
   if diff2.sum() < curr_min:
      curr_min = diff2.sum()
      min = i
print(curr_min,min)

sigma = 88.75*0.00217278
df_day_comb_long['call_price'] = black_scholes(S, K, T=250/365, r=0, sigma = sigma, option="C")
diff = df_day_comb_long['call_price'] - df_day_comb_long['mid_price_ccnc']
diff2 = diff**2
diff2.sum()

df = df_day_comb_long
labels = ['call_price','mid_price_ccnc']
plt.plot(df[['new_ts']], df[labels])
plt.legend(labels)
plt.show()

df_day_comb_long['diff'] = df_day_comb_long['call_price'] - df_day_comb_long['mid_price_ccnc']

strin = 'diff'
print(f"mean:{np.mean(df_day_comb_long[strin])};median{np.median(df_day_comb_long[strin])};std{np.std(df_day_comb_long[strin])}")

sns.distplot(df_day_comb_long[strin], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()


# Pre-calculated lookup table (limited range for example)
cdf_table = {}
for z in range(-300, 301):  # Adjust range as needed
  z_score = z / 100
  cdf_value = norm.cdf(z_score)
  cdf_table[z_score] = cdf_value

cdf_table
# Example usage
z_score = 1  # Example value within the table range
if z_score in cdf_table:
  cdf_value = cdf_table[z_score]
  print("Pre-calculated CDF for z-score", z_score, ":", cdf_value)
else:
  print("Value outside pre-calculated range. Use 'get_cdf' function for any value.")

cdf_table = {-2.0: 0.0227501319481792, -1.99: 0.02329546775021182, -1.98: 0.023851764341508513, -1.97: 0.024419185280222543, -1.96: 0.024997895148220435, -1.95: 0.02558805952163861, -1.94: 0.026189844940452685, -1.93: 0.02680341887705495, -1.92: 0.027428949703836802, -1.91: 0.0280666066597725, -1.9: 0.028716559816001803, -1.89: 0.02937898004040943, -1.88: 0.03005403896119979, -1.87: 0.030741908929465954, -1.86: 0.0314427629807527, -1.85: 0.03215677479561371, -1.84: 0.03288411865916389, -1.83: 0.0336249694196283, -1.82: 0.03437950244588998, -1.81: 0.03514789358403879, -1.8: 0.03593031911292579, -1.79: 0.03672695569872628, -1.78: 0.03753798034851679, -1.77: 0.03836357036287122, -1.76: 0.03920390328748265, -1.75: 0.040059156863817086, -1.74: 0.040929508978807365, -1.73: 0.04181513761359493, -1.72: 0.04271622079132892, -1.71: 0.0436329365240319, -1.7: 0.04456546275854304, -1.69: 0.045513977321549805, -1.68: 0.046478657863720046, -1.67: 0.04745968180294733, -1.66: 0.04845722626672282, -1.65: 0.0494714680336481, -1.64: 0.05050258347410371, -1.63: 0.051550748490089365, -1.62: 0.05261613845425206, -1.61: 0.053698928148119655, -1.6: 0.054799291699557974, -1.59: 0.05591740251946941, -1.58: 0.057053433237754185, -1.57: 0.05820755563855299, -1.56: 0.05937994059479304, -1.55: 0.06057075800205901, -1.54: 0.061780176711811886, -1.53: 0.06300836446397844, -1.52: 0.06425548781893582, -1.51: 0.0655217120889165, -1.5: 0.06680720126885806, -1.49: 0.06811211796672544, -1.48: 0.06943662333333173, -1.47: 0.07078087699168552, -1.46: 0.07214503696589378, -1.45: 0.07352925960964836, -1.44: 0.07493369953432708, -1.43: 0.07635850953673913, -1.42: 0.07780384052654638, -1.41: 0.07926984145339239, -1.4: 0.08075665923377107, -1.39: 0.08226443867766892, -1.38: 0.08379332241501436, -1.37: 0.08534345082196698, -1.36: 0.08691496194708503, -1.35: 0.08850799143740196, -1.34: 0.09012267246445244, -1.33: 0.09175913565028077, -1.32: 0.09341750899347179, -1.31: 0.09509791779523902, -1.3: 0.09680048458561036, -1.29: 0.09852532904974787, -1.28: 0.10027256795444206, -1.27: 0.10204231507481915, -1.26: 0.1038346811213004, -1.25: 0.10564977366685535, -1.24: 0.10748769707458694, -1.23: 0.10934855242569191, -1.22: 0.11123243744783456, -1.21: 0.11313944644397739, -1.2: 0.11506967022170822, -1.19: 0.11702319602310873, -1.18: 0.11900010745520073, -1.17: 0.12100048442101824, -1.16: 0.12302440305134343, -1.15: 0.12507193563715036, -1.14: 0.1271431505627983, -1.13: 0.12923811224001786, -1.12: 0.1313568810427307, -1.11: 0.13349951324274717, -1.1: 0.13566606094638267, -1.09: 0.1378565720320355, -1.08: 0.140071090088769, -1.07: 0.14230965435593923, -1.06: 0.1445722996639096, -1.05: 0.1468590563758959, -1.04: 0.1491699503309814, -1.03: 0.15150500278834367, -1.02: 0.15386423037273478, -1.01: 0.15624764502125466, -1.0: 0.15865525393145707, -0.99: 0.1610870595108309, -0.98: 0.16354305932769236, -0.97: 0.16602324606352958, -0.96: 0.16852760746683781, -0.95: 0.17105612630848183, -0.94: 0.1736087803386246, -0.93: 0.1761855422452579, -0.92: 0.17878637961437172, -0.91: 0.1814112548917972, -0.9: 0.18406012534675947, -0.89: 0.18673294303717264, -0.88: 0.18942965477671214, -0.87: 0.19215020210369615, -0.86: 0.1948945212518084, -0.85: 0.19766254312269238, -0.84: 0.20045419326044966, -0.83: 0.2032693918280684, -0.82: 0.20610805358581302, -0.81: 0.2089700878716016, -0.8: 0.2118553985833967, -0.79: 0.21476388416363712, -0.78: 0.21769543758573312, -0.77: 0.22064994634264962, -0.76: 0.22362729243759938, -0.75: 0.2266273523768682, -0.74: 0.22964999716479056, -0.73: 0.2326950923008974, -0.72: 0.23576249777925118, -0.71: 0.2388520680899867, -0.7: 0.24196365222307303, -0.69: 0.24509709367430943, -0.68: 0.24825223045357053, -0.67: 0.25142889509531013, -0.66: 0.25462691467133614, -0.65: 0.2578461108058647, -0.64: 0.26108629969286157, -0.63: 0.26434729211567753, -0.62: 0.26762889346898305, -0.61: 0.27093090378300566, -0.6: 0.2742531177500736, -0.59: 0.27759532475346493, -0.58: 0.28095730889856435, -0.57: 0.28433884904632417, -0.56: 0.28773971884902705, -0.55: 0.29115968678834636, -0.54: 0.294598516215698, -0.53: 0.29805596539487644, -0.52: 0.3015317875469662, -0.51: 0.3050257308975194, -0.5: 0.3085375387259869, -0.49: 0.31206694941739055, -0.48: 0.31561369651622256, -0.47: 0.3191775087825558, -0.46: 0.32275811025034773, -0.45: 0.32635522028791997, -0.44: 0.32996855366059363, -0.43: 0.3335978205954577, -0.42: 0.3372427268482495, -0.41: 0.3409029737723226, -0.4: 0.3445782583896758, -0.39: 0.3482682734640176, -0.38: 0.3519727075758372, -0.37: 0.3556912451994533, -0.36: 0.35942356678200876, -0.35: 0.3631693488243809, -0.34: 0.36692826396397193, -0.33: 0.37069998105934643, -0.32: 0.37448416527667994, -0.31: 0.3782804781779807, -0.3: 0.3820885778110474, -0.29: 0.3859081188011227, -0.28: 0.3897387524442028, -0.27: 0.3935801268019605, -0.26: 0.3974318867982395, -0.25: 0.4012936743170763, -0.24: 0.40516512830220414, -0.23: 0.40904588485799415, -0.22: 0.41293557735178543, -0.21: 0.4168338365175577, -0.2: 0.42074029056089696, -0.19: 0.42465456526520456, -0.18: 0.42857628409909926, -0.17: 0.4325050683249616, -0.16: 0.4364405371085672, -0.15: 0.4403823076297575, -0.14: 0.44432999519409355, -0.13: 0.44828321334543886, -0.12: 0.45224157397941617, -0.11: 0.4562046874576832, -0.1: 0.460172162722971, -0.09: 0.4641436074148279, -0.08: 0.4681186279860126, -0.07: 0.47209682981947887, -0.06: 0.47607781734589316, -0.05: 0.4800611941616275, -0.04: 0.48404656314716926, -0.03: 0.48803352658588733, -0.02: 0.492021686283098, -0.01: 0.4960106436853684, 0.0: 0.5, 0.01: 0.5039893563146316, 0.02: 0.5079783137169019, 0.03: 0.5119664734141126, 0.04: 0.5159534368528308, 0.05: 0.5199388058383725, 0.06: 0.5239221826541068, 0.07: 0.5279031701805211, 0.08: 0.5318813720139874, 0.09: 0.5358563925851721, 0.1: 0.539827837277029, 0.11: 0.5437953125423168, 0.12: 0.5477584260205839, 0.13: 0.5517167866545611, 0.14: 0.5556700048059064, 0.15: 0.5596176923702425, 0.16: 0.5635594628914329, 0.17: 0.5674949316750384, 0.18: 0.5714237159009007, 0.19: 0.5753454347347955, 0.2: 0.579259709439103, 0.21: 0.5831661634824423, 0.22: 0.5870644226482146, 0.23: 0.5909541151420059, 0.24: 0.5948348716977958, 0.25: 0.5987063256829237, 0.26: 0.6025681132017605, 0.27: 0.6064198731980396, 0.28: 0.6102612475557972, 0.29: 0.6140918811988774, 0.3: 0.6179114221889526, 0.31: 0.6217195218220193, 0.32: 0.6255158347233201, 0.33: 0.6293000189406536, 0.34: 0.6330717360360281, 0.35: 0.6368306511756191, 0.36: 0.6405764332179913, 0.37: 0.6443087548005467, 0.38: 0.6480272924241628, 0.39: 0.6517317265359823, 0.4: 0.6554217416103242, 0.41: 0.6590970262276774, 0.42: 0.6627572731517505, 0.43: 0.6664021794045423, 0.44: 0.6700314463394064, 0.45: 0.67364477971208, 0.46: 0.6772418897496523, 0.47: 0.6808224912174442, 0.48: 0.6843863034837774, 0.49: 0.6879330505826095, 0.5: 0.6914624612740131, 0.51: 0.6949742691024806, 0.52: 0.6984682124530338, 0.53: 0.7019440346051236, 0.54: 0.705401483784302, 0.55: 0.7088403132116536, 0.56: 0.712260281150973, 0.57: 0.7156611509536759, 0.58: 0.7190426911014356, 0.59: 0.7224046752465351, 0.6: 0.7257468822499265, 0.61: 0.7290690962169943, 0.62: 0.732371106531017, 0.63: 0.7356527078843225, 0.64: 0.7389137003071384, 0.65: 0.7421538891941353, 0.66: 0.7453730853286639, 0.67: 0.7485711049046899, 0.68: 0.7517477695464294, 0.69: 0.7549029063256906, 0.7: 0.758036347776927, 0.71: 0.7611479319100133, 0.72: 0.7642375022207488, 0.73: 0.7673049076991025, 0.74: 0.7703500028352095, 0.75: 0.7733726476231317, 0.76: 0.7763727075624006, 0.77: 0.7793500536573503, 0.78: 0.7823045624142668, 0.79: 0.7852361158363629, 0.8: 0.7881446014166034, 0.81: 0.7910299121283983, 0.82: 0.7938919464141869, 0.83: 0.7967306081719316, 0.84: 0.7995458067395503, 0.85: 0.8023374568773076, 0.86: 0.8051054787481916, 0.87: 0.8078497978963038, 0.88: 0.8105703452232879, 0.89: 0.8132670569628273, 0.9: 0.8159398746532405, 0.91: 0.8185887451082028, 0.92: 0.8212136203856283, 0.93: 0.8238144577547422, 0.94: 0.8263912196613754, 0.95: 0.8289438736915182, 0.96: 0.8314723925331622, 0.97: 0.8339767539364704, 0.98: 0.8364569406723077, 0.99: 0.8389129404891691, 1.0: 0.8413447460685429, 1.01: 0.8437523549787453, 1.02: 0.8461357696272652, 1.03: 0.8484949972116563, 1.04: 0.8508300496690187, 1.05: 0.8531409436241041, 1.06: 0.8554277003360904, 1.07: 0.8576903456440608, 1.08: 0.859928909911231, 1.09: 0.8621434279679645, 1.1: 0.8643339390536173, 1.11: 0.8665004867572528, 1.12: 0.8686431189572693, 1.13: 0.8707618877599821, 1.14: 0.8728568494372018, 1.15: 0.8749280643628496, 1.16: 0.8769755969486566, 1.17: 0.8789995155789818, 1.18: 0.8809998925447993, 1.19: 0.8829768039768913, 1.2: 0.8849303297782918, 1.21: 0.8868605535560226, 1.22: 0.8887675625521654, 1.23: 0.8906514475743081, 1.24: 0.8925123029254131, 1.25: 0.8943502263331446, 1.26: 0.8961653188786995, 1.27: 0.8979576849251809, 1.28: 0.8997274320455579, 1.29: 0.9014746709502521, 1.3: 0.9031995154143897, 1.31: 0.904902082204761, 1.32: 0.9065824910065282, 1.33: 0.9082408643497193, 1.34: 0.9098773275355476, 1.35: 0.911492008562598, 1.36: 0.913085038052915, 1.37: 0.914656549178033, 1.38: 0.9162066775849856, 1.39: 0.917735561322331, 1.4: 0.9192433407662289, 1.41: 0.9207301585466077, 1.42: 0.9221961594734536, 1.43: 0.9236414904632608, 1.44: 0.925066300465673, 1.45: 0.9264707403903516, 1.46: 0.9278549630341062, 1.47: 0.9292191230083144, 1.48: 0.9305633766666683, 1.49: 0.9318878820332746, 1.5: 0.9331927987311419, 1.51: 0.9344782879110836, 1.52: 0.9357445121810641, 1.53: 0.9369916355360216, 1.54: 0.9382198232881881, 1.55: 0.939429241997941, 1.56: 0.940620059405207, 1.57: 0.941792444361447, 1.58: 0.9429465667622459, 1.59: 0.9440825974805306, 1.6: 0.945200708300442, 1.61: 0.9463010718518804, 1.62: 0.9473838615457479, 1.63: 0.9484492515099107, 1.64: 0.9494974165258963, 1.65: 0.9505285319663519, 1.66: 0.9515427737332772, 1.67: 0.9525403181970526, 1.68: 0.9535213421362799, 1.69: 0.9544860226784502, 1.7: 0.955434537241457, 1.71: 0.9563670634759681, 1.72: 0.957283779208671, 1.73: 0.9581848623864051, 1.74: 0.9590704910211927, 1.75: 0.9599408431361829, 1.76: 0.9607960967125173, 1.77: 0.9616364296371288, 1.78: 0.9624620196514833, 1.79: 0.9632730443012737, 1.8: 0.9640696808870742, 1.81: 0.9648521064159612, 1.82: 0.9656204975541101, 1.83: 0.9663750305803717, 1.84: 0.9671158813408361, 1.85: 0.9678432252043863, 1.86: 0.9685572370192473, 1.87: 0.9692580910705341, 1.88: 0.9699459610388002, 1.89: 0.9706210199595906, 1.9: 0.9712834401839981, 1.91: 0.9719333933402275, 1.92: 0.9725710502961632, 1.93: 0.973196581122945, 1.94: 0.9738101550595473, 1.95: 0.9744119404783614, 1.96: 0.9750021048517795, 1.97: 0.9755808147197774, 1.98: 0.9761482356584915, 1.99: 0.9767045322497881, 2.0: 0.9772498680518208}







norm.cdf(0)

min_multip = 0
currdiffsum = math.inf
for intmultip in range(9363000,9364000,1):
    multip = float(intmultip/10000000.00000000)
    df_day_comb_long[f'should_be_ccn{multip}'] = df_day_comb_long['mid_price_ccn']*(multip)+df_day_comb_long['mid_price_ccnc']
    df_day_comb_long['diff'] = (df_day_comb_long[f'should_be_ccn{multip}'] - 10000)**2
    diffsum = df_day_comb_long['diff'].sum()
    if diffsum < currdiffsum:
       print(multip)
       currdiffsum = diffsum
       min_multip = multip
print(min_multip)
multip = 0.9363913

min_multip = 0
currdiffsum = math.inf
for intmultip in range(1063400,1063500,1):
    multip = float(intmultip/1000000.00000000)
    df_day_comb_long[f'should_be_ccn{multip}']= df_day_comb_long['mid_price_ccn']*(multip)-df_day_comb_long['mid_price_ccnc']
    df_day_comb_long['diff'] = (df_day_comb_long[f'should_be_ccn{multip}'] - 10000)**2
    diffsum = df_day_comb_long['diff'].sum()
    if diffsum < currdiffsum:
       print(multip)
       currdiffsum = diffsum
       min_multip = multip
print(min_multip)
min_multip = 1.063471

df_day_comb_long[['should_be_ccn0.936','mid_price_ccn']]
min_multip
a = 1000/1000000000000000.00000000
f'{a:.100f}'
min_multip
currdiffsum
df_day_comb_long
multip = float(999/10000.00000000)
multip = float()


multip = 0.9363913
multip = 1.063471
df_day_comb_long[f'should_be_ccn']= df_day_comb_long['mid_price_ccn']*(multip)-df_day_comb_long['mid_price_ccnc']

df_day_comb_long[f'should_be_ccn{multip}']= 10000*((multip)**((250*1000000-df_day_comb_long['new_ts'])/(250*1000000)))+df_day_comb_long['mid_price_ccnc']

df_day_comb_long[f'should_be_ccn']= (10000-df_day_comb_long['mid_price_ccnc'])/(multip)

df_day_comb_long[f'should_be_ccn']= (10000-df_day_comb_long['mid_price_ccnc'])/(multip)

df_day_comb_long[f'should_be_ccnc']= (10000)-(df_day_comb_long['mid_price_ccn']*(multip))

df_day_comb_long[['timeval','mid_price_ccnc','mid_price_ccn']]
df_day_comb_long['10000-ccnc']= (10000)-(df_day_comb_long['mid_price_ccnc'])


df_day_comb_long['a'] = abs(df_day_comb_long['10000-ccnc'] - 9700)

df_day_comb_long['b'] = df_day_comb_long['mid_price_ccn'] - 9700

df_day_comb_long['x'] = (10000 + df_day_comb_long['mid_price_ccnc'])/df_day_comb_long['mid_price_ccn']


df = df_day_comb_long
labels = ['x']
plt.plot(df[['new_ts']], df[labels])
plt.legend(labels)
plt.show()

df = df_day_comb_long
labels = ['should_be_ccnc','mid_price_ccnc']
plt.plot(df[['new_ts']], df[labels])
plt.legend(labels)
plt.show()


fig, axs = plt.subplots()
color = 'tab:red'
#ax1.set_xlabel('time (s)')
axs.set_ylabel('ccn-ccnc', color=color)
axs.plot(df_day_comb_long['new_ts'], df_day_comb_long['ccn-ccnc'], color=color)
axs.tick_params(axis='y', labelcolor=color)
ax2 = axs.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('mid_price_ccn', color=color)  # we already handled the x-label with ax1
ax2.plot(df_day_comb_long['new_ts'], df_day_comb_long['mid_price_ccn'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


fig, axs = plt.subplots(3)
color = 'tab:red'
#ax1.set_xlabel('time (s)')
#ax1.set_ylabel('exp', color=color)
axs[0].plot(df_day1_long['timestamp_ccn'], df_day1_long['mid_price_ccn'], color=color)
#ax1.tick_params(axis='y', labelcolor=color)
ax2 = axs[0].twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
#ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(df_day1_long['timestamp_ccn'], df_day1_long['mid_price_ccnc'], color=color)
#ax2.tick_params(axis='y', labelcolor=color)
color = 'tab:red'
#ax1.set_xlabel('time (s)')
#ax1.set_ylabel('exp', color=color)
axs[1].plot(df_day2_long['timestamp_ccn'], df_day2_long['mid_price_ccn'], color=color)
#ax1.tick_params(axis='y', labelcolor=color)
ax2 = axs[1].twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
#ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(df_day2_long['timestamp_ccn'], df_day2_long['mid_price_ccnc'], color=color)
#ax2.tick_params(axis='y', labelcolor=color)
color = 'tab:red'
#ax1.set_xlabel('time (s)')
#ax1.set_ylabel('exp', color=color)
axs[2].plot(df_day3_long['timestamp_ccn'], df_day3_long['mid_price_ccn'], color=color)
#ax1.tick_params(axis='y', labelcolor=color)
ax2 = axs[2].twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
#ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(df_day3_long['timestamp_ccn'], df_day3_long['mid_price_ccnc'], color=color)
#ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



df_day_combined_long.reset_index()
df_day_combined_long['mid_price_gbb'] = df_day_combined_long['mid_price_cc']*4 + df_day_combined_long['mid_price_sb']*6 +df_day_combined_long['mid_price_rs']*1
df_day_combined_long['mid_price_rss'] = df_day_combined_long['mid_price_gb'] - df_day_combined_long['mid_price_cc']*4 - df_day_combined_long['mid_price_sb']*6 
df_day_combined_long['mid_price_ccc'] = (df_day_combined_long['mid_price_gb'] - df_day_combined_long['mid_price_rs'] - df_day_combined_long['mid_price_sb']*6)/4 
df_day_combined_long['mid_price_sbb'] = (df_day_combined_long['mid_price_gb'] - df_day_combined_long['mid_price_rs'] - df_day_combined_long['mid_price_cc']*4)/6 

df_day_combined_long['norm_mid_price_gb'] = df_day_combined_long['mid_price_gb']/df_day_combined_long['mid_price_gb'].max()
df_day_combined_long['norm_mid_price_rs'] = df_day_combined_long['mid_price_rs']/df_day_combined_long['mid_price_rs'].max()
df_day_combined_long['norm_mid_price_cc'] = df_day_combined_long['mid_price_cc']/df_day_combined_long['mid_price_cc'].max()
df_day_combined_long['norm_mid_price_sb'] = df_day_combined_long['mid_price_sb']/df_day_combined_long['mid_price_sb'].max()

df_day_combined_long['norm_mid_price_rscc'] = df_day_combined_long['norm_mid_price_rs']+df_day_combined_long['norm_mid_price_cc']
df_day_combined_long['norm_mid_price_rssb'] = df_day_combined_long['norm_mid_price_rs']+df_day_combined_long['norm_mid_price_sb']
df_day_combined_long['norm_mid_price_sbcc'] = df_day_combined_long['norm_mid_price_sb']+df_day_combined_long['norm_mid_price_cc']


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

strin = 'diff_rs'
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

strin = 'diff_gb'
print(f"mean:{np.mean(df_day_combined_long[strin])};median{np.median(df_day_combined_long[strin])};std{np.std(df_day_combined_long[strin])}")

sns.distplot(df_day_combined_long[strin], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()

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














from scipy.stats import norm
from scipy.optimize import root_scalar

def black_scholes(S, K, T, sigma, r, option="C"):
  """
  Calculates the Black-Scholes option price.

  Args:
      S (float): Spot price of the underlying asset.
      K (float): Strike price of the option.
      T (float): Time to expiration in years.
      sigma (float): Volatility of the underlying asset.
      r (float): Risk-free interest rate.
      option (str, optional): Option type ("C" for call, "P" for put). Defaults to "C".

  Returns:
      float: The Black-Scholes option price.
  """

  d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  if option == "C":
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
  elif option == "P":
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
  else:
    raise ValueError("Invalid option type. Please specify 'C' for call or 'P' for put.")

  return price

# Function to find the stock price (S) that results in the target option price
def find_stock_price(target_price, K, T, sigma, r, option="C"):
  """
  Uses root_scalar to find S that results in the target option price.

  Args:
      target_price (float): The known option price.
      K (float): Strike price of the option.
      T (float): Time to expiration in years.
      sigma (float): Volatility of the underlying asset.
      r (float): Risk-free interest rate.
      option (str, optional): Option type ("C" for call, "P" for put). Defaults to "C".

  Returns:
      float (or None): The estimated stock price (S) or None if not found.
  """

  def f(S):
    return black_scholes(S, K, T, sigma, r) - target_price

  # Initial guess for stock price (can affect convergence)
  S_guess = 10000

  try:
    sol = root_scalar(f, x0=S_guess)
    return sol.root
  except (RuntimeError, ValueError):
    print("Warning: Could not find a solution for stock price. Consider revising initial guess or checking model assumptions.")
    return None



# Example usage
target_price = df_day_comb_long['mid_price_ccnc']  # Known option price
K = 10000  # Strike price
T = 250/365  # Time to expiration (in years)
sigma =  88.75*0.00217278 # Volatility
r = 0  # Risk-free interest rate
option = "C"  # Call option


estimated_stock_price = []
for i in range(len(target_price)):
  estimated_stock_price.append(find_stock_price(target_price[i], K, T, sigma, r, option))

if estimated_stock_price:
  print("Estimated stock price based on Black-Scholes formula:", estimated_stock_price)
else:
  print("Estimation failed. See warnings for details.")

find_stock_price(637.5, K, T, sigma, r, option)

df_day_comb_long['should_be_ccn'] = pd.DataFrame(estimated_stock_price)

df = df_day_comb_long
labels = ['should_be_ccn','mid_price_ccn']
plt.plot(df[['new_ts']], df[labels])
plt.legend(labels)
plt.show()

df = df_day_comb_long
labels = ['mid_price_ccnc','mid_price_ccn']
plt.plot(df[['new_ts']], df[labels])
plt.legend(labels)
plt.show()

t = df_day_comb_long[['new_ts']]
data1 = df_day_comb_long[['mid_price_ccnc']]
data2 = df_day_comb_long[['mid_price_ccn']]

fig, ax1 = plt.subplots()

color = 'tab:red'
#ax1.set_xlabel('time (s)')
#ax1.set_ylabel('exp', color=color)
ax1.plot(t, data1, color=color)
#ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
#ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
#ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


