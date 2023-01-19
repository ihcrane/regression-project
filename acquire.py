from env import get_connection
import pandas as pd
import os


def get_zillow_data():
    
    '''
    This function is used to get zillow data from sql database.
    '''
    test = '%'
    if os.path.isfile('zillow.csv'):
        
        return pd.read_csv('zillow.csv')
    
    else:
        
        url = get_connection('zillow')
        query = f'''
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
		        taxvaluedollarcnt, yearbuilt, fips, lotsizesquarefeet, transactiondate 
                FROM properties_2017
                LEFT JOIN predictions_2017 USING(id)
                WHERE propertylandusetypeid = 261 AND transactiondate LIKE '2017{test}{test}';
                '''
        df = pd.read_sql(query, url)
        df.to_csv('zillow.csv')
        return df

def wrangle_zillow():
    
    '''
    This function is used to get zillow data from sql database, renaming columns, 
    dropping nulls and duplicates.
    '''
    
    df = get_zillow_data()
    
    # renaming columns
    df = df.rename(columns={'bedroomcnt':'bed',
                        'bathroomcnt':'bath',
                        'calculatedfinishedsquarefeet':'sqft',
                        'taxvaluedollarcnt':'tax_value',
                        'yearbuilt':'year',
                        'lotsizesquarefeet':'lot_sqft'})
    
    # drop Unnamed: 0 column
    df = df.drop(columns=['Unnamed: 0', 'transactiondate'])

    #drop nulls
    df = df.dropna()
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df