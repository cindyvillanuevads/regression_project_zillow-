import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr


def plot_categorical_and_continuous_vars (df, categorical, continuous):
    '''
    takes in a df, a list of categorical columns, list
    '''
    print('Discrete with Continuous')
    plt.figure(figsize=(13, 6))
    for cat in categorical:
        for cont in continuous:
            sns.boxplot(x= cat, y=cont, data=df)
            plt.show()
            sns.swarmplot(x=cat, y=cont, data=df)
            plt.show()
    print('Continuous with Continuous')        
    sns.pairplot(df[continuous], kind="reg", plot_kws={'line_kws':{'color':'red'}}, corner=True)
    return



def heatmap (df):
    '''
    Takes in a df and return a heatmap
    '''
    x = len(list(df.select_dtypes(exclude='O').columns))
    df_corr= df.corr()
    sns.heatmap(df_corr, cmap='Purples', annot=True, linewidth=0.5, mask= np.triu(df_corr))
    plt.ylim(0, x)
    return



def distribution_single_var (df, columns):
    '''
    Take in a train_df and return a distributions of single varibles
    '''

    for col in columns:
            #plot
            plt.show()
            plt.figure(figsize=(10, 6))
            sns.displot(df[col])
            plt.title(col)
            plt.show()

    return


def plot_variable_pairs(df, target):
    '''
    Takes in a dataframe and a target and returns  plots of all the pairwise relationships 
    along with the regression line for each pair.
    '''
    
    # get the list of the columns  that are not object type
    columns = list(df.select_dtypes(exclude= 'O').columns)
    #remove target from columns
    columns.remove(target)
    
    #plot
    for col in columns:
        sns.lmplot(x= col, y= target, data=df, line_kws={'color': 'red'})
        plt.title(col)
        plt.show()
    return