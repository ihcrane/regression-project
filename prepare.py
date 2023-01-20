import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression



def train_val_test(df, target=None, stratify=None, seed=42):
    
    '''Split data into train, validate, and test subsets with 60/20/20 ratio'''
    
    train, val_test = train_test_split(df, train_size=0.6, random_state=seed)
    
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    return train, val, test
    

def x_y_split(df, target, seed=42):
    
    '''
    This function is used to split train, val, test into X_train, y_train, X_val, y_val, X_train, y_test
    '''
    
    train, val, test = train_val_test(df, target, seed)
    
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_val = val.drop(columns=[target])
    y_val = val[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_val, y_val, X_test, y_test

def mm_scaler(train, val, test, col_list):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Min Max Scaler and out puts
    scaled data for all three data splits
    '''
    # calls the Min Max Scaler function and fits to train data
    mm_scale = MinMaxScaler()
    mm_scale.fit(train[col_list])
    
    # transforms all three data sets
    mm_train = mm_scale.transform(train[col_list])
    mm_val = mm_scale.transform(val[col_list])
    mm_test = mm_scale.transform(test[col_list])
    
    # converts all three transformed data sets to pandas DataFrames
    mm_train = pd.DataFrame(mm_train, columns=col_list)
    mm_val = pd.DataFrame(mm_val, columns=col_list)
    mm_test = pd.DataFrame(mm_test, columns=col_list)
    
    return mm_train, mm_val, mm_test

def ss_scaler(train, val, test, col_list):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Standard Scaler and out puts
    scaled data for all three data splits
    '''

    # calls Standard Scaler function and fits to train data
    ss_scale = StandardScaler()
    ss_scale.fit(train[col_list])
    
    # transforms all three data sets
    ss_train = ss_scale.transform(train[col_list])
    ss_val = ss_scale.transform(val[col_list])
    ss_test = ss_scale.transform(test[col_list])
    
    # converts all three transformed data sets to pandas Data Frames
    ss_train = pd.DataFrame(ss_train, columns=col_list)
    ss_val = pd.DataFrame(ss_val, columns=col_list)
    ss_test = pd.DataFrame(ss_test, columns=col_list)
    
    return ss_train, ss_val, ss_test

def rs_scaler(train, val, test, col_list):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Robust Scaler and out puts
    scaled data for all three data splits
    '''

    # calls Robust Scaler funtion and fits to train data set
    rs_scale = RobustScaler()
    rs_scale.fit(train[col_list])
    
    # transforms all three data sets
    rs_train = rs_scale.transform(train[col_list])
    rs_val = rs_scale.transform(val[col_list])
    rs_test = rs_scale.transform(test[col_list])
    
    # converts transformed data into pandas Data Frames
    rs_train = pd.DataFrame(rs_train, columns=col_list)
    rs_val = pd.DataFrame(rs_val, columns=col_list)
    rs_test = pd.DataFrame(rs_test, columns=col_list)
    
    return rs_train, rs_val, rs_test

def qt_scaler(train, val, test, col_list, dist='normal'):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Quantile Transformer and out puts
    scaled data for all three data splits
    '''

    # calls Quantile Transformer function and fits to train data set
    qt_scale = QuantileTransformer(output_distribution=dist, random_state=42)
    qt_scale.fit(train[col_list])
    
    # transforms all three data sets
    qt_train = qt_scale.transform(train[col_list])
    qt_val = qt_scale.transform(val[col_list])
    qt_test = qt_scale.transform(test[col_list])
    
    # converts all three transformed data sets to pandas Data Frames
    qt_train = pd.DataFrame(qt_train, columns=col_list)
    qt_val = pd.DataFrame(qt_val, columns=col_list)
    qt_test = pd.DataFrame(qt_test, columns=col_list)
    
    return qt_train, qt_val, qt_test


def remove_outliers(df, k=1.5):

    '''
    This function is to remove the top 25% and bottom 25% of the data for each column.
    This removes the top and bottom 50% for every column to ensure all outliers are gone.
    '''
    a=[]
    b=[]
    fences=[a, b]
    features= []
    col_list = []
    i=0
    for col in df:
            new_df=np.where(df[col].nunique()>8, True, False)
            if new_df==True:
                if df[col].dtype == 'float' or df[col].dtype == 'int':
                    '''
                    for each feature find the first and third quartile
                    '''
                    q1, q3 = df[col].quantile([.25, .75])
                    '''
                    calculate inter quartile range
                    '''
                    iqr = q3 - q1
                    '''
                    calculate the upper and lower fence
                    '''
                    upper_fence = q3 + (k * iqr)
                    lower_fence = q1 - (k * iqr)
                    '''
                    appending the upper and lower fences to lists
                    '''
                    a.append(upper_fence)
                    b.append(lower_fence)
                    '''
                    appending the feature names to a list
                    '''
                    features.append(col)
                    '''
                    assigning the fences and feature names to a dataframe
                    '''
                    var_fences= pd.DataFrame(fences, columns=features, index=['upper_fence', 'lower_fence'])
                    
                    col_list.append(col)
                else:
                    print(col)
                    print('column is not a float or int')
            else:
                print(f'{col} column ignored')
    '''
    for loop used to remove the data deemed unecessary 
    '''
    for col in col_list:
        df = df[(df[col]<= a[i]) & (df[col]>= b[i])]
        i+=1
    return df, var_fences

def plot_categorical_and_continuous_vars(train, cont_vars, cat_vars):
    train_corr = train[cont_vars].corr(method='spearman')
    sns.heatmap(train_corr)
    plt.show()
    
    for col in cont_vars:
        sns.lmplot(x='tax_value', y=col, data=train.sample(1000))
        plt.show()
    
    for col in cat_vars:
        sns.stripplot(x=col, y='tax_value', data=train.sample(1000))
        plt.show()


def plot_variable_pairs(train):
    sns.pairplot(data=train.sample(2000), diag_kind='hist', kind='reg')
    plt.show()
    

def rmse(preds, target):
    return sqrt(mean_squared_error(preds['actual'], preds[target]))


def select_kbest(df, cont, cat, y, k):
    
    '''
    This function takes a data frame, a list of continuous variables, a list of categorical variables,
    the target variable, and top number of features wanted. It scales the continuous variables and 
    creates X_train and y_train data frames. It then creates dummies for the categorical variables. 
    After all the data has been manipulated it runs the SelectKBest for f_regression and returns 
    the top k number of variables.
    '''
    
    # fitting and scaling the continuous variables
    mms = MinMaxScaler()
    df[cont] = mms.fit_transform(df[cont])
    
    # creating X_train and y_train data frames
    X_df_scaled = df.drop(columns=[y])
    y_df = df[y]
    
    # creating dummies for the categorical variables
    X_df_scaled = pd.get_dummies(X_df_scaled, columns=cat)
    
    # fitting the regression model to the data
    f_selector = SelectKBest(f_regression, k=k)
    f_selector.fit(X_df_scaled, y_df)
    
    # determining which variables are the top k variables
    f_select_mask = f_selector.get_support()
    
    # returning data frame of the only the top k variables
    return X_df_scaled.iloc[:,f_select_mask]


def rfe(df, cont, cat, y, k):
    
    '''
    This function takes a data frame, a list of continuous variables, a list of categorical variables,
    the target variable, and top number of features wanted. It scales the continuous variables and 
    creates X_train and y_train data frames. It then creates dummies for the categorical variables.
    The function then runs the RFE function using linear regression to determine which features are best.
    It returns a data frame with each features and the ranking for the user to determine which features
    they want to use.
    '''
    
    # fitting and scaling the continuous variables
    mms = MinMaxScaler()
    df[cont] = mms.fit_transform(df[cont])
    
    # creating X_train and y_train data frames
    X_df_scaled = df.drop(columns=[y])
    y_df = df[y]
    
    # creating dummies for the categorical variables
    X_df_scaled = pd.get_dummies(X_df_scaled, columns=cat)
        
    # creating linear regressiong RFE model based on k number
    lm = LinearRegression()
    model = RFE(lm, n_features_to_select=k)
    
    # fitting model to scaled data
    model.fit(X_df_scaled, y_df)
    
    # determine rankings for each feature
    ranks = model.ranking_
    columns = X_df_scaled.columns.tolist()
    
    # creating data frame of ranking and column names
    feature_ranks = pd.DataFrame({'ranking':ranks,
                                  'feature':columns})
    
    # returns created data frame of feature rankings
    return feature_ranks.sort_values('ranking')