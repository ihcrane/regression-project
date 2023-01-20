#!/usr/bin/env python
# coding: utf-8

# In[1]:


from acquire import wrangle_zillow
from prepare import remove_outliers, x_y_split, rmse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


# In[7]:


df = wrangle_zillow()


# In[8]:


df, var_fences = remove_outliers(df)


# In[9]:


def fips_plot(x, y):
    sns.boxplot(x=x, y=y,data=df)
    plt.xlabel('FIPS Code')
    plt.ylabel('Tax Value')
    plt.title('Tax Value in each county')
    plt.show()


# In[10]:


fips_plot('fips', 'tax_value')


# In[11]:


cont = ['bed','bath','sqft','year','lot_sqft']
cat = ['fips']
target = 'tax_value'


# In[22]:


def testing_corr():
    for col in cont:
        corr, p = stats.pearsonr(df[col], df['tax_value'])
        print(f'The correlation between {col} and tax_value is: {corr:.3f}')
        print('--------------------------------------------------------')


# In[ ]:




