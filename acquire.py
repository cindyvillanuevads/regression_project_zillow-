import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")


# *************************************  connection url **********************************************

# Create helper function to get the necessary connection url.
def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    from env import host, username, password
    return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'

# ************************************ generic acquire function***************************************************************


def get_data_from_sql(db_name, query):
    """
    This function takes in a string for the name of the database I want to connect to
    and a query to obtain my data from the Codeup server and return a DataFrame.
    db_name : df name in a string type
    query: aalready created query that was named as query 
    Example:
    query = '''
    SELECT * 
    FROM table_name;
    '''
    df = get_data_from_sql('telco_churn', query)
    """
    df = pd.read_sql(query, get_connection(db_name))
    return df





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


# ****************************************************************************************************************************************
#  this function was shared by a classmate. I add some things
#*******************************************************************************************************************************\

def miss_dup_values(df):
    '''
    this function takes a dataframe as input and will output metrics for missing values and duplicated rows, 
    and the percent of that column that has missing values and duplicated rows
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        #total of duplicated
    dup = df.duplicated().sum()  
        # Percentage of missing values
    dup_percent = 100 * dup / len(df)
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.")
    print( "  ")
    print (f"** There are {dup} duplicate rows that represents {round(dup_percent, 2)}% of total Values**")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns



