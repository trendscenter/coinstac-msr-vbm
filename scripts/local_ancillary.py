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