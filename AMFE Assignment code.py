#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# In[2]:


dt= pd.read_csv('C:/Users/rohns/OneDrive/Desktop/AMFE project/FLRG Historical Data.csv')


# In[3]:


df= dt.copy()
df.head()


# In[6]:


df.info()


# In[5]:


df['DCOILBRENTEU']=pd.to_numeric(df['DCOILBRENTEU'], errors='coerce')
df['Date']= pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')


# In[7]:


df['Price'].plot()


# In[33]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], label='Price', color='black')
plt.title("Franklin Green Bond Time Series Plot", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("ETF Price", fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True)
plt.show()


# In[8]:


df['DCOILBRENTEU'].plot()


# In[34]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Oil Price'], label='Price', color='black')
plt.title("Brent Oil Time Series Plot", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True)
plt.show()


# In[9]:


df= df.rename(columns={'DCOILBRENTEU':'Oil Price'})


# In[10]:


df['% Change Price']= (df['Price']/df['Price'].shift(1))-1
df['% Change Oil Price']= (df['Oil Price']/ df['Oil Price'].shift(1))-1


# In[11]:


df= df.dropna()


# In[12]:


df['% Change Price'].plot()


# In[37]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['% Change Price'], label='Price', color='black')
plt.title("Franklin Green Bond first difference Time Plot", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("returns", fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True)
plt.show()


# In[13]:


df['% Change Oil Price'].plot()


# In[36]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['% Change Oil Price'], label='Price', color='black')
plt.title("Brent Oil First Differened Plot", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("returns", fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True)
plt.show()


# In[14]:


# Augmented Dickey-Fuller Test
def adf_test(series, title=""):
    print(f"Results of ADF Test for {title}")
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"Test Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"# Lags Used: {result[2]}")
    print(f"Number of Observations: {result[3]}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")
    print("\n")
    return result


# In[15]:


adf_test(df['Price'], "NAV")
adf_test(df['% Change Price'], "NAV Change%")
adf_test(df['Oil Price'], "EU_brentOIL")
adf_test(df['% Change Oil Price'], "EU_brentOIL % Change")


# In[ ]:



clean_df= df.iloc[:,1:3]
clean_df.head()


# In[17]:


#Johansen Cointegration Test

johansen_test = coint_johansen(clean_df, det_order=0, k_ar_diff=1)

# Print eigenvalues
print("Eigenvalues:")
print(johansen_test.eig)

# Print the trace statistic
print("\nTrace Statistic:")
print(johansen_test.lr1)

# Print critical values for the trace statistic
print("\nCritical Values (90%, 95%, 99%):")
print(johansen_test.cvt)

# Check the maximum eigenvalue statistic
print("\nMax-Eigenvalue Statistic:")
print(johansen_test.lr2)

# Print critical values for the maximum eigenvalue statistic
print("\nCritical Values for Max-Eigenvalue (90%, 95%, 99%):")
print(johansen_test.cvm)


# In[43]:


#Correlation Heatmap
sns.heatmap(df[['Oil Price', 'Price']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Plot')
plt.show()


# In[19]:


clean_df_change= df.iloc[:,3:]
clean_df_change.head()


# In[ ]:





# In[ ]:





# In[20]:


from statsmodels.tsa.api import VAR
model = VAR(clean_df_change)

order_selection = model.select_order(maxlags=6)
print("Optimal Lag Order:", order_selection.selected_orders)


# In[39]:


optimal_lag = order_selection.selected_orders['hqic']  # Example using AIC
var_model = model.fit(optimal_lag)
print(var_model.summary())


# In[22]:


# Residuals
residuals = var_model.resid
print(residuals)


# In[23]:


from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals.values)
print("Durbin-Watson Test:", dw)


# In[24]:


# Jarque-Bera Test
for col in residuals.columns:
    jb = jarque_bera(residuals[col])
    print(f"Jarque-Bera Test for {col}: {jb}")


# In[40]:


# Granger Causality Test
granger_Price = var_model.test_causality('% Change Price', '% Change Oil Price', kind='f')
granger_Oil_Price = var_model.test_causality('% Change Oil Price', '% Change Price', kind='f')
print("Granger Causality Price -> Oil Price:", granger_Price.summary())
print("Granger Causality Oil Price-> Price:", granger_Oil_Price.summary())


# In[26]:


# Impulse Response Function
irf = var_model.irf(20)
irf.plot(orth=False)
plt.show()


# In[27]:


# Forecast Error Variance Decomposition (FEVD)
fevd = var_model.fevd(50)
fevd.plot()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




