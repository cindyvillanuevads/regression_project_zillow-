import pandas as pd
import numpy as np
import os
import acquire 

from sklearn.model_selection import train_test_split
# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")



# ***************************************************************************************************
#                                     ZILLOW DB
# ***************************************************************************************************

#acquire data for the first time
def get_new_zillow():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with columns :
     bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
    '''
    sql_query = '''
    SELECT parcelid, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,
    taxvaluedollarcnt
    FROM properties_2017
    JOIN predictions_2017 as pred USING (parcelid)
    WHERE pred.transactiondate >= '2017-05-01' AND pred.transactiondate <= '2017-08-31'
    AND propertylandusetypeid > 259 AND propertylandusetypeid  < 266;
    '''
    return pd.read_sql(sql_query, get_connection('zillow'))

#acquire data main function 
def get_zillow():
    '''
    This function reads in telco_churn data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = get_new_zillow()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    return df





def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')                                  
    return train, validate, test

def split_Xy (train, validate, test, target):
    '''
    This function takes in three dataframe (train, validate, test) and a target  and splits each of the 3 samples
    into a dataframe with independent variables and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    Example:
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_Xy (train, validate, test, 'Fertility' )
    '''
    
    #split train
    X_train = train.drop(columns= [target])
    y_train= train[target]
    #split validate
    X_validate = validate.drop(columns= [target])
    y_validate= validate[target]
    #split validate
    X_test = test.drop(columns= [target])
    y_test= test[target]

    print(f'X_train -> {X_train.shape}               y_train->{y_train.shape}')
    print(f'X_validate -> {X_validate.shape}         y_validate->{y_validate.shape} ')        
    print(f'X_test -> {X_test.shape}                  y_test>{y_test.shape}') 
    return  X_train, y_train, X_validate, y_validate, X_test, y_test



def wrangle_zillow():
    ''''
    This function will acquire zillow db using get_new_zillow function. then it will use another
    function named  clean_zillwo that drops duplicates,  nulls, all houses that do not have bedrooms and bathrooms,
    houses that calculatedfinishedsquarefeet < 800.
     bedroomcnt, yearbuilt, fips are changed to int.
    return cleaned zillow DataFrame
    '''
    df = acquire.get_new_zillow()
    zillow_df = clean_zillow(df)
    return zillow_df




