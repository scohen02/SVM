'''
Wesleyan University, COMP 343, Spring 2022
Final Project
Lexie Silverman, Sophie Cohen
'''

import numpy as np
import pandas as pd


########################## Data helper functions #############################

def split_data(df, train_proportion):
    ''' Inputs
            * df: dataframe containing data
            * train_proportion: proportion of data in df that will be used for
                training. 1-train_proportion is proportion of data to be used
                for testing
        Output
            * train_df: dataframe containing training data
            * test_df: dataframe containing testing data
    '''
    # Make sure there are row numbers
    df = df.reset_index(drop=True)

    # Reorder examples and split data according to train proportion
    train = df.sample(frac=train_proportion, axis=0)
    test = df.drop(index=train.index)
    return train, test

def divide_k_folds(df, num_folds):
    ''' Inputs
            * df: dataframe containing data
            * num_folds: number of folds
        Output
            * folds: lists of folds, each fold is subset of df dataframe
    '''
    folds = []
    for subset in np.array_split(df, num_folds):
        folds.append(subset)
    return folds

def get_X_y_data(df, features, target):
    ''' Split dataframe into X and y numpy arrays '''
    X = np.array([np.array(x) for _, x in df[features].iterrows()])
    y = np.array(df[target])
    return X, y

def sgn(prod):
    if prod >= 0:
        return 1
    else:
        return -1
