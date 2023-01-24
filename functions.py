#!/usr/bin/env python
# coding: utf-8

# In[26]:


from acquire import wrangle_zillow
from prepare import remove_outliers, x_y_split, rmse
import functions as f

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


# In[27]:


def fips_plot(x, y, df):
    
    '''
    This function creates a boxplot of the value spread for each county.
    '''
    df_1 = df
    df_1['fips'] = df_1['fips'].map({6059: 'Orange',6037:'Los Angeles',6111:'Ventura'})
    
    sns.boxplot(x=x, y=y,data=df_1)
    plt.xlabel('County')
    plt.ylabel('Property Value')
    plt.title('Tax Value in each county')
    plt.show()


# In[28]:


def anova_test(df, col):
    
    '''
    This function runs and ANOVA stats test on the col listed.
    '''
    
    if col == 'fips':
        la_county = df[df['fips']==6037]['tax_value']
        orange_county = df[df['fips']==6059]['tax_value']
        ventura_county = df[df['fips']==6111]['tax_value']
        
        f, p = stats.f_oneway(la_county, orange_county, ventura_county)
    else:
        bed_2 = df[df['bed']==2]['tax_value']
        bed_3 = df[df['bed']==3]['tax_value']
        bed_4 = df[df['bed']==4]['tax_value']
        bed_5 = df[df['bed']==5]['tax_value']
        
        f, p = stats.f_oneway(bed_2, bed_3, bed_4, bed_5)
                    
    
    alpha = .05
    
    if p < alpha:
        print("We reject the null.")
    else:
        print("We fail to reject the null.")


# In[29]:


def testing_corr(df, col):
    
    '''
    This function tests the correlation of each continuous feature compared to the tax_value.
    '''
    
    alpha = .05
    
    corr, p = stats.pearsonr(df[col], df['tax_value'])
    print(f'The correlation between {col} and property value is: {corr:.3f}')
    print('--------------------------------------------------------')
    if p < alpha:
        print("We reject the null.")
    else:
        print("We fail to reject the null.")


# In[30]:


def county_plot(df):
    
    '''
    This function creates a histogram of the values of properties in each county.
    '''
    
    la_county = df[df['fips']=='Los Angeles']['tax_value']
    orange_county = df[df['fips']=='Orange']['tax_value']
    ventura_county = df[df['fips']=='Ventura']['tax_value']
    
    plt.hist(x = la_county, color = 'yellow', alpha = .4, edgecolor = 'black', label = 'Los Angeles')
    plt.hist(x = orange_county, color = 'red', alpha = .5, edgecolor = 'black', label = 'Orange')
    plt.hist(x = ventura_county, color = 'blue', alpha = .5, edgecolor = 'black', label = 'Ventura')
    plt.xlabel('Property Value')
    plt.ylabel('Number of Properties')
    plt.title('Comparing the Value of Properties in Each County')
    plt.legend()
    plt.show()


# In[31]:


def beds_plot(df):
    
    '''
    This function creates a bar chart for the average property value dependent on the number of bedrooms.
    '''
    
    fig, ax = plt.subplots()
    bplot = sns.barplot(x='bed', y='tax_value', data=df)
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Property Value')
    plt.title('Average Property Value per Number of Bedrooms')
    ax.bar_label(bplot.containers[0], padding= 6)
    plt.show()


# In[32]:


def baths_plot(df):
    
    '''
    This function creates a bar chart for the average property value dependent on the number of bathrooms.
    '''
    
    fig, ax = plt.subplots()
    bplot = sns.barplot(x='bath', y='tax_value', data=df)
    plt.xlabel('Number of Bathrooms')
    plt.ylabel('Property Value')
    plt.title('Average Property Value per Number of Bathrooms')
    ax.bar_label(bplot.containers[0], padding= 14)
    plt.show()


# In[33]:


def sqft_plot(df):
    
    '''
    This function creates a scatter plot of the comparison between square feet and property value with a 
    line through that shows the correlation between the two variables.
    '''
    
    sns.regplot(x='sqft', y='tax_value', data=df.sample(2000), line_kws={'color':'red'})
    plt.xlabel('Square Feet')
    plt.ylabel('Property Value')
    plt.title('Property Value for Amount of Square Feet')
    plt.show()


# In[34]:


def lot_sqft_plot(df):
    
    '''
    This function creates a scatter plot of the comparison between lot square feet and property value with a 
    line through that shows the correlation between the two variables.
    '''
    
    sns.regplot(x='lot_sqft', y='tax_value', data=df.sample(2000),line_kws={'color':'red'})
    plt.xlabel('Lot Square Feet')
    plt.ylabel('Property Value')
    plt.title('Property Value for Amount of Square Feet in the Lot')
    plt.show()


# In[35]:


def year_built_plot(df):
    
    '''
    This function creates a scatter plot of the comparison between year built and property value with a 
    line through that shows the correlation between the two variables.
    '''
    
    sns.regplot(x='year', y='tax_value', data=df.sample(2000), line_kws={'color':'red'})
    plt.xlabel('Year Property Built')
    plt.ylabel('Property Value')
    plt.title('Property Value for Year Property Built')
    plt.show()


# # Dropping columns, getting dummies and scaling data

# In[36]:


def prep_zillow(df):
    
    '''
    This function preps the zillow data by dropping the year and lot_sqft columns. It then gets dummies
    for the fips, bed and bath columns. Splits the data into X_train, y_train, X_val, y_val, X_test and y_test.
    Then it scales the necessary data and returns the variables.
    '''
    
    # dropping year and lot_sqft due to lack of use in data
    df.drop(columns=['year', 'lot_sqft'], inplace = True)
    
    # getting dummies of the fips, bed and bath columns
    df = pd.get_dummies(df, columns=['fips', 'bed','bath'])
    
    # splitting data into train, val and test
    X_train, y_train, X_val, y_val, X_test, y_test = x_y_split(df, 'tax_value')
    
    # initializing MinMaxScaler
    mms = MinMaxScaler()

    # scaling X_train, X_val, and X_test data sets
    X_train[['sqft']] = mms.fit_transform(X_train[['sqft']])
    X_val[['sqft']] = mms.transform(X_val[['sqft']])
    X_test[['sqft']] = mms.transform(X_test[['sqft']])
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# In[37]:


def preds_table(y_train):
    
    '''
    This function creates the predictions data frame with the actual data and baseline prediciton.
    '''
    
    preds = pd.DataFrame({'actual':y_train,
                          'baseline':y_train.mean()})
    
    # creating baseline rmse
    baseline_rmse = rmse(preds, 'baseline')
    
    print(f'The RMSE for the baseline is: {baseline_rmse:.2f}')
    
    return preds, baseline_rmse


# In[38]:


def lr_testing(X_train, y_train, preds):
    
    '''
    This function takes the X_train, y_train and preds table. It creates and fits the LinearRegression model
    to the train data set and predicts based off the X_train data. It appends those predctions to the preds
    table and returns the table along with the RMSE of those predictions.
    '''
    
    # initializing LinearRegression model
    lm = LinearRegression()

    # fitting model to train data set
    lm.fit(X_train, y_train)
    
    # making predictions and calculating RMSE
    preds['lm_preds'] = lm.predict(X_train)
    lm_rmse = rmse(preds, 'lm_preds')
    
    print(f'The RMSE for the Linear Regression Model on the train data set is: {lm_rmse:.2f}')
    
    return preds, lm_rmse


# In[39]:


def lasso_testing(X_train, y_train, preds):
    
    '''
    This function takes the X_train, y_train and preds table. It creates and fits the LassoLars model
    to the train data set and predicts based off the X_train data. It appends those predctions to the preds
    table and returns the table along with the RMSE of those predictions.
    '''
    
    # initializing LassoLars model
    lasso = LassoLars(alpha=.1)

    # fitting model to train data set
    lasso.fit(X_train, y_train)
    
    # making predictions and calculating RMSE
    preds['lasso_preds'] = lasso.predict(X_train)
    lasso_rmse = rmse(preds, 'lasso_preds')
    
    print(f'The RMSE for the Lasso Lars Model on the train data set is: {lasso_rmse:.2f}')
    
    return preds, lasso_rmse


# In[40]:


def poly_testing(X_train, y_train, preds):
    
    '''
    This function takes the X_train, y_train and preds table. It creates and fits the PolynomialFeatures model
    to the train data set and transforms the train data. It then creates and fits the LinearRegressiong model
    to the newly transformed data and predicts based off the transformed train data. It appends those 
    predctions to the preds table and returns the table along with the RMSE of those predictions.
    '''
    
    # initializing PolynomialFeatures
    pf = PolynomialFeatures(degree=2)

    # fitting and transforming train data set
    pf.fit(X_train, y_train)
    X_polynomial = pf.transform(X_train)
    
    # initializing LinearRegression odel
    lmtwo = LinearRegression()
    
    # fitting model to transformed train data set
    lmtwo.fit(X_polynomial, y_train)
    
    # making predictions and calculating RMSE
    preds['poly_preds'] = lmtwo.predict(X_polynomial)
    poly_rmse = rmse(preds, 'poly_preds')
    
    print(f'The RMSE for the Polynomial Linear Regression Model on the train data set is: {poly_rmse:.2f}')
    
    return preds, poly_rmse


# In[41]:


def lasso_poly_testing(X_train, y_train, preds):
    
    '''
    This function takes the X_train, y_train and preds table. It creates and fits the PolynomialFeatures model
    to the train data set and transforms the train data. It then creates and fits the LassoLars model
    to the newly transformed data and predicts based off the transformed train data. It appends those 
    predctions to the preds table and returns the table along with the RMSE of those predictions.
    '''
    
    # initializing Polynomial Features
    pf = PolynomialFeatures(degree=2)

    # fitting and transforming train data set
    pf.fit(X_train, y_train)
    X_polynomial = pf.transform(X_train)
    
    # initializing LassoLars model
    lassotwo = LassoLars(alpha=.1)

    # fitting to transformed train data set
    lassotwo.fit(X_polynomial, y_train)

    # making predictions and calculating RMSE
    preds['lasso_poly'] = lassotwo.predict(X_polynomial)
    lassopoly_rmse = rmse(preds, 'lasso_poly')
    
    print(f'The RMSE for the Polynomial Lasso Lars Model on the train data set is: {lassopoly_rmse:.2f}')
    
    return preds, lassopoly_rmse


# In[42]:


def tweedie_norm_testing(X_train, y_train, preds):
    
    '''
    This function takes the X_train, y_train and preds table. It creates and fits the TweedieRegressor model
    to the train data set and predicts based off the X_train data. It appends those predctions to the preds
    table and returns the table along with the RMSE of those predictions.
    '''
    
    # initializing TweedieRegressor
    tweedie = TweedieRegressor(power=0)

    # fitting to train data set
    tweedie.fit(X_train, y_train)
    
    # making predictions and calculating RMSE
    preds['tweedie'] = tweedie.predict(X_train)
    tweedie_rmse = rmse(preds, 'tweedie')
    
    print(f'The RMSE for the Tweedie Regressor Model on the train data set is: {tweedie_rmse:.2f}')
    
    return preds, tweedie_rmse


# In[43]:


def models_rmse(baseline_rmse, lm_rmse, lasso_rmse, poly_rmse, lassopoly_rmse, tweedie_rmse):
    
    '''
    This function takes the rmse of all the tested models and makes a data frame from them.
    '''
    rmse_df = pd.DataFrame({'model':['baseline','linear', 'lasso','linear_poly', 'lasso_poly','tweedie_norm'],
                            'rmse':[baseline_rmse, lm_rmse, lasso_rmse, poly_rmse, lassopoly_rmse, tweedie_rmse]})
    return rmse_df


# In[44]:


def model_plot(rmse_df):
    
    '''
    This function takes the rmse data frame and plots all of the models on a bar chart.
    '''
    
    fig, ax = plt.subplots(figsize=(10,7))
    bplot = sns.barplot(x='model',y='rmse', data=rmse_df.sort_values('rmse'))
    plt.ylabel('RMSE')
    plt.xlabel('Model')
    plt.title('RMSE for Each Tested Model')
    plt.ylim(0, 250000)
    ax.bar_label(bplot.containers[0], padding= 6)


# In[45]:


def val_preds(y_train, y_val):
    
    '''
    This function creates a val preds table from the y_val data and the baseline of the y_train data.
    '''
    
    val_preds = pd.DataFrame({'actual':y_val,
                              'baseline':y_train.mean()})
    return val_preds


# In[46]:


def val_poly_test(X_train, y_train, X_val, y_val, val_preds):
    
    '''
    This function takes the X_train, y_train, X_val, y_val and val_preds table. It creates and fits the 
    PolynomialFeatures model to the train data set and transforms the train and val data. It then creates 
    and fits the LinearRegression model to the newly transformed data and predicts based off the transformed 
    val data. It appends those predctions to the val_preds table and returns the table along with the RMSE of 
    those predictions.
    '''
    
    # initializing Polynomial Features
    pf = PolynomialFeatures(degree=2)

    # fitting to train data set and transforming train and val data set
    pf.fit(X_train, y_train)
    X_poly = pf.transform(X_train)
    X_val_poly = pf.transform(X_val)
    
    # initializing LinearRegression model
    lmtwo = LinearRegression()

    # fitting to transformed data set
    lmtwo.fit(X_poly, y_train)
    
    # making predictions on val data set and calculating RMSE
    val_preds['poly_preds'] = lmtwo.predict(X_val_poly)
    poly_rmse = rmse(val_preds, 'poly_preds')
    
    print(f'The RMSE for the Polynomial Linear Regression Model on the validate data set is: {poly_rmse:.2f}')
    
    return val_preds, poly_rmse


# In[47]:


def val_lasso_test(X_train, y_train, X_val, y_val, val_preds):
    
    '''
    This function takes the X_train, y_train, X_val, y_val and val_preds table. It creates and fits the 
    LassoLars model to the train data set and transforms the train and val data. It appends those predctions 
    to the val_preds table and returns the table along with the RMSE of those predictions.
    '''
    
    # initilizing LassoLars model
    lasso = LassoLars(alpha=.1)

    # fitting to train data set
    lasso.fit(X_train, y_train)
    
    # making predictions on the val data set and calculating RMSE
    val_preds['lasso_preds'] = lasso.predict(X_val)
    lasso_rmse = rmse(val_preds, 'lasso_preds')
    
    print(f'The RMSE for the Lasso Lars Model on the validate data set is: {lasso_rmse:.2f}')
    
    return val_preds, lasso_rmse


# In[48]:


def val_lassopoly_test(X_train, y_train, X_val, y_val, val_preds):
    
    '''
    This function takes the X_train, y_train, X_val, y_val and val_preds table. It creates and fits the 
    PolynomialFeatures model to the train data set and transforms the train and val data. It then creates 
    and fits the LassoLars model to the newly transformed data and predicts based off the transformed 
    val data. It appends those predctions to the val_preds table and returns the table along with the RMSE of 
    those predictions.
    '''
    
    # initializing Polynomial Features
    pf = PolynomialFeatures(degree=2)

    # fitting to train data set and transforming train and val data 
    pf.fit(X_train, y_train)
    X_poly = pf.transform(X_train)
    X_val_poly = pf.transform(X_val)
    
    # initializing LassoLars model
    lassotwo = LassoLars(alpha=.1)

    # fitting to transformed train data
    lassotwo.fit(X_poly, y_train)

    # making predictions on val data set and calculating RMSE
    val_preds['lasso_poly'] = lassotwo.predict(X_val_poly)
    lassopoly_rmse = rmse(val_preds, 'lasso_poly')
    
    print(f'The RMSE for the Polynomial Lasso Lars Model on the validate data set is: {lassopoly_rmse:.2f}')
    
    return val_preds, lassopoly_rmse


# In[49]:


def val_models_rmse(baseline_rmse, lasso_rmse, poly_rmse, lassopoly_rmse):
    
    '''
    This function takes the rmse values of the tested models and creates a data frame from the data.
    '''
    
    val_rmse_df = pd.DataFrame({'model':['baseline', 'lasso','linear_poly', 'lasso_poly'],
              'rmse':[baseline_rmse, lasso_rmse, poly_rmse, lassopoly_rmse]})
    
    return val_rmse_df


# In[50]:


def test_poly(X_train, y_train, X_test, y_test):
    
    '''
    This function takes the X_train, y_train, X_test, and y_test. It creates and fits the PolynomialFeatures 
    model to the train data set and transforms the train and test data. It then creates and fits the 
    LassoLars model to the newly transformed data and predicts based off the transformed val data. 
    It uses those predctions to create the test_preds table and returnes the RMSE of those predictions.
    '''
    
    # initializing Polynomial Features
    pf = PolynomialFeatures(degree=2)

    # fitting to train data set and transforming train and test data sets
    pf.fit(X_train, y_train)
    X_poly = pf.transform(X_train)
    X_test_poly = pf.transform(X_test)
    
    # initializing LassoLars model
    lassotwo = LassoLars(alpha=.1)

    # fitting to transformed train data set
    lassotwo.fit(X_poly, y_train)
    
    # making predictions on the test data set
    test_preds = pd.DataFrame({'actual':y_test,
                               'test_pred':lassotwo.predict(X_test_poly)})
    
    # calculate and return RMSE
    test = rmse(test_preds, 'test_pred')
    
    print(f'The RMSE for the Polynomial Lasso Lars Model on the test dataset is: {test:.2f}')


# In[ ]:




