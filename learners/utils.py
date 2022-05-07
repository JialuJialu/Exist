import numpy as np
import pandas as pd
from copy import copy
import pdb
import math

#-----------------------Loading data from CSV files-----------------------------

def load_expected_post(df, features):
    header = list(df.columns)
    for feature in features:
        # for load_next_iter, this step is handled by combine_data in cegis
        # 2.72 comes from the natural logarithm e, the columns 1 and 2.72 are 
        # are added to do linear regression with intercept
        if feature == "1" or feature == "2.72" or feature not in list(df.columns):
            df[feature] = df.eval(feature, engine="python")
    X = np.array(df[features], dtype=np.float32)
    y = np.array(df['post-init'], dtype=np.float32) 
    return X, y

def load_next_iter(df_init, df_next, features):
    for feature in features:
        if feature not in list(df_init.columns):
            df_init[feature] = df_init.eval(feature, engine="python")
        if feature not in list(df_next.columns):
            df_next[feature] = df_next.eval(feature, engine="python")
    X = np.array(df_init[features], dtype=np.float32)
    pre = np.array(df_init["pre"], dtype=np.float32)
    post_cur = np.array(df_init["post"], dtype=np.float32)
    Y = np.array(df_next[features], dtype=np.float32)
    post_next = np.array(df_next["post"], dtype=np.float32)
    G_next = np.array(df_next["G"], dtype=np.float32)
    try:
        assert (len(Y))%(len(X)) == 0
    except AssertionError:
        pdb.set_trace()
    Y = np.array_split(Y, len(X))
    post_next = np.array_split(post_next, len(X))
    G_next = np.array_split(G_next, len(X))
    return X, pre, post_cur, Y, G_next, post_next

# ------Handling numerical errors when exponentiating or taking logarithms------
def log_no_nan(X):
    X[np.where(X<=0)] = 1e-8
    X = np.log(X)
    return X

def exp_no_nan(X):
    X = np.exp(X)
    X[np.isinf(X)] = (10 ** 8)
    return X


def exp_int_no_nan(x):
    x = np.exp(x)
    if np.isinf(x): 
        x = 10**8
    return min(10**8, x)