import coinstacparsers
from coinstacparsers import parsers
import warnings
import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm

def get_cost(y_actual, y_predicted):
    return np.average((y_actual-y_predicted)**2)

def check_cols_to_normalize(X):
    columns_to_normalize=[]
    max_vals = X.max(axis=0).to_numpy()
    minval = np.min(max_vals[np.nonzero(max_vals)])
    X_headers = list(X.columns)
    ranges = max_vals / minval
    temp_cols_indxs = np.where(ranges > 10000)[0]
    for col_indx in temp_cols_indxs:
        columns_to_normalize.append(X_headers[col_indx])
    return columns_to_normalize

def normalize_columns(data_df, cols):
    for col in cols:
        data_df[col] = (data_df[col] - data_df[col].mean())/(data_df[col].std())
    return data_df