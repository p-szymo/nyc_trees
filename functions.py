import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

def yes_to_one(df, cols):
    '''Turn columns with 'Yes' and 'No' values into 1s and 0s.
    Overwrites the input columns!
    '''
    for col in cols:
        df[col] = np.where(df[col] == 'Yes', 1, 0)
        
def find_extremes(df):
    '''Takes in a dataframe and returns a list of columns with values farther than 4 standard deviations from the mean.'''
    extreme_list = []
    for column in list(df.columns):
        if df[column].max() > (df[column].mean() + 4*df[column].std()):
            extreme_list.append(column)
        if df[column].min() < (df[column].mean() - 4*df[column].std()):
            extreme_list.append(column)
    return extreme_list

def rein_extremes(df, columns):
    '''Takes in a dataframe and a list of columns and changes any values farther than 4 standard deviations from the mean
    to 4 standard deviations from the mean.
    Overwrites the input column!'''
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        conditions = [df[column] > mean + 4*std,
                      df[column] < mean - 4*std]
        choices = [mean + 4*std,
                   mean - 4*std]
        df[column] = np.select(conditions, choices, df[column])
        
def two_way_tests(series_list):
    '''Takes in a list of series and runs a two-sided t-test on every combination within the list.
    Returns a dictionary with the indices of the tested series as the keys and the test results as the values.
    '''
    compare_dict = {}
    for i in range(len(series_list)):
        count = i+1
        while count < len(series_list):
            compare_dict.update({(i,count): ttest_ind(series_list[i], series_list[count])})
            count += 1
    return compare_dict

def two_way_tests_dicts(series_dict):
    '''Takes in a list of series and runs a two-sided t-test on every combination within the list.
    Returns a dictionary with the indices of the tested series as the keys and the test results as the values.
    '''
    series_list = list(series_dict.values())
    compare_dict = {}
    for i in range(len(series_list)):
        count = i+1
        while count < len(series_list):
            compare_dict.update({(list(series_dict.keys())[i],list(series_dict.keys())[count]): ttest_ind(series_list[i], series_list[count])})
            count += 1
    return compare_dict