#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation
"""
import json
import numpy as np
import sys
import regression as reg
import warnings
from parsers import vbm_parser
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


def local_0(args):
    input_list = args["input"]
    lamb = input_list["lambda"]

    (X, y) = vbm_parser(args)

    y = pd.DataFrame(
        y.loc[:, 0:4])  # comment this line to demonstrate docker hanging
    y_labels = ['{}_{}'.format('voxel', str(i)) for i in y.columns]

    computation_output_dict = {
        "output": {
            "computation_phase": "local_0"
        },
        "cache": {
            "covariates": X.values.tolist(),
            "dependents": y.values.tolist(),
            "lambda": lamb,
            "y_labels": y_labels,
        },
    }

    return json.dumps(computation_output_dict)


def local_1(args):
    """Read data from the local sites, perform local regressions and send
    local statistics to the remote site"""

    X = args["cache"]["covariates"]
    y = args["cache"]["dependents"]
    lamb = args["cache"]["lambda"]
    y_labels = args["cache"]["y_labels"]
    y = pd.DataFrame(y, columns=y_labels)

    biased_X = sm.add_constant(X)
    meanY_vector, lenY_vector = [], []

    local_params = []
    local_sse = []
    local_pvalues = []
    local_tvalues = []
    local_rsquared = []

    for column in y.columns:
        curr_y = list(y[column])
        meanY_vector.append(np.mean(curr_y))
        lenY_vector.append(len(y))

        # Printing local stats as well
        model = sm.OLS(curr_y, biased_X.astype(float)).fit()
        local_params.append(model.params)
        local_sse.append(model.ssr)
        local_pvalues.append(model.pvalues)
        local_tvalues.append(model.tvalues)
        local_rsquared.append(model.rsquared_adj)

    keys = ["beta", "sse", "pval", "tval", "rsquared"]
    local_stats_list = []
    for index, _ in enumerate(y_labels):
        values = [
            local_params[index].tolist(), local_sse[index],
            local_pvalues[index].tolist(), local_tvalues[index].tolist(),
            local_rsquared[index]
        ]
        local_stats_dict = {key: value for key, value in zip(keys, values)}
        local_stats_list.append(local_stats_dict)

    # +++++++++++++++++++++ Adding site covariate columns +++++++++++++++++++++
    site_covar_list = args["input"]["site_covar_list"]

    site_matrix = np.zeros(
        (np.array(X).shape[0], len(site_covar_list)), dtype=int)
    site_df = pd.DataFrame(site_matrix, columns=site_covar_list)

    select_cols = [
        col for col in site_df.columns if args["state"]["clientId"] in col
    ]

    site_df[select_cols] = 1
    biased_X = np.concatenate((biased_X, site_df.values), axis=1)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    beta_vec_size = biased_X.shape[1]

    computation_output = {
        "output": {
            "beta_vec_size": beta_vec_size,
            "number_of_regressions": len(y_labels),
            "computation_phase": "local_1"
        },
        "cache": {
            "beta_vec_size": beta_vec_size,
            "number_of_regressions": len(y_labels),
            "covariates": biased_X.tolist(),
            "dependents": y.values.tolist(),
            "lambda": lamb,
            "y_labels": y_labels,
            "mean_y_local": meanY_vector,
            "count_local": lenY_vector,
            "local_stats_list": local_stats_list
        }
    }

    return json.dumps(computation_output)


def local_2(args):

    X = args["cache"]["covariates"]
    y = args["cache"]["dependents"]
    lamb = args["cache"]["lambda"]

    beta_vec_size = args["cache"]["beta_vec_size"]
    number_of_regressions = args["cache"]["number_of_regressions"]

    mask_flag = args["input"].get("mask_flag",
                                  np.zeros(number_of_regressions, dtype=bool))

    biased_X = np.array(X)
    y = pd.DataFrame(y)

    w = args["input"]["remote_beta"]

    gradient = np.zeros((number_of_regressions, beta_vec_size))

    for i in range(number_of_regressions):
        y_ = y[i]
        w_ = w[i]
        if not mask_flag[i]:
            gradient[i, :] = (
                1 / len(X)) * np.dot(biased_X.T, np.dot(biased_X, w_) - y_)

    computation_phase = {
        "cache": {
            "covariates": X,
            "dependents": y.values.tolist(),
            "lambda": lamb,
            "y_labels": args["cache"]["y_labels"],
            "mean_y_local": args["cache"]["mean_y_local"],
            "count_local": args["cache"]["count_local"],
            "local_stats_list": args["cache"]["local_stats_list"],
        },
        "output": {
            "local_grad": gradient.tolist(),
            "computation_phase": "local_2"
        }
    }

    return json.dumps(computation_phase)


def local_3(args):
    cache_list = args["cache"]
    X = cache_list["covariates"]
    y = cache_list["dependents"]
    y_labels = cache_list["y_labels"]
    lamb = cache_list["lambda"]

    computation_output = {
        "output": {
            "mean_y_local": args["cache"]["mean_y_local"],
            "count_local": args["cache"]["count_local"],
            "local_stats_list": args["cache"]["local_stats_list"],
            "y_labels": y_labels,
            "computation_phase": 'local_3'
        },
        "cache": {
            "covariates": X,
            "dependents": y,
            "lambda": lamb
        }
    }

    return json.dumps(computation_output)


def local_4(args):
    """Computes the SSE_local, SST_local and varX_matrix_local

    Args:
        args (dictionary): {"input": {
                                "avg_beta_vector": ,
                                "mean_y_global": ,
                                "computation_phase":
                                },
                            "cache": {
                                "covariates": ,
                                "dependents": ,
                                "lambda": ,
                                "dof_local": ,
                                }
                            }

    Returns:
        computation_output (json): {"output": {
                                        "SSE_local": ,
                                        "SST_local": ,
                                        "varX_matrix_local": ,
                                        "computation_phase":
                                        }
                                    }

    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local

    """
    cache_list = args["cache"]
    input_list = args["input"]

    X = cache_list["covariates"]
    y = cache_list["dependents"]
    biased_X = np.array(X)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    y = pd.DataFrame(y)
    SSE_local, SST_local = [], []
    for index, column in enumerate(y.columns):
        curr_y = y[column].values
        SSE_local.append(
            reg.sum_squared_error(biased_X, curr_y, avg_beta_vector))
        SST_local.append(
            np.sum(
                np.square(np.subtract(curr_y, mean_y_global[index])),
                dtype=float))

    varX_matrix_local = np.dot(biased_X.T, biased_X)

    computation_output = {
        "output": {
            "SSE_local": SSE_local,
            "SST_local": SST_local,
            "varX_matrix_local": varX_matrix_local.tolist(),
            "computation_phase": "local_4"
        },
        "cache": {}
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.argv[1])
    phase_key = list(reg.listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_0(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_0' in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_1' in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_2a' in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_2b' in phase_key:
        computation_output = local_3(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_3' in phase_key:
        computation_output = local_4(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
