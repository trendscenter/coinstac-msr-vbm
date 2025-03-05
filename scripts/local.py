#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation
"""
import json
import numpy as np
import sys
import os
import warnings
from coinstacparsers import parsers
import pandas as pd
import local_ancillary as lc
from regression import listRecursive, sum_squared_error, y_estimate
from utils import log
from nipype_utils import average_nifti

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


def local_0(args):
    input_list = args["input"]
    lamb = input_list["lambda"]
    mask = os.path.join('/computation', 'assets', 'mask_6mm.nii')
    (X, y) = parsers.vbm_parser(args, mask)

    """
    X: 
        df(
            column: [age, isControl, sex_M], 
            rows:[28, 1,1], 
            index={M02108714_swc1t1avg_6mm.nii, M02110676_swc1t1avg_6mm.nii, ...}
        )
    y:
        df(
            column: [voxel_0, voxel_1, voxel_2, voxe_3, ..., voxel_9955], 
            rows=[[0.101961, 0.082353,..., 0.01230]]
            index={0, 1, 2, 3, 4, ....}
        )
    y_labels: 
        [voxel_voxel_1, voxel_voxel_2, ...., voxel_voxel_9955]
    """
    columns_to_normalize = lc.check_cols_to_normalize(X)
    #y = pd.DataFrame(
    #    y.loc[:, 0:24])  # comment this line to demonstrate docker hanging
    y_labels = ['{}_{}'.format('voxel', str(i)) for i in y.columns]


    """average nifti computation"""
    # covar_x = average_nifti(args)
    # lc.to_csv(covar_x, os.path.join(cache_dir, 'X_df'))

    tol = input_list["tol"]
    eta = input_list["eta"]

    # raise Exception(X, y, args, y_labels)
    computation_output_dict = {
        "output": {
            "computation_phase": "local_0",
            "columns_to_normalize": columns_to_normalize,
            "avg_nifti": "avg_nifti.nii",
            "tol": tol,
            "eta": eta
        },
        "cache": {
            "covariates": X.values.tolist(),
            "dependents": y.values.tolist(),
            "lambda": lamb,
            "y_labels": y_labels,
            "X_labels": list(X.columns)
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
    X_labels = args['cache']['X_labels']
    y = pd.DataFrame(y, columns=y_labels)

    input_list = args['input']
    X = lc.normalize_columns(X, input_list["columns_to_normalize"])
    log(f'\n\nNormalizing the following column values to their z-scores: {input_list["columns_to_normalize"]} \n ', args['state'])

    meanY_vector, lenY_vector, local_stats_list, beta_vector = (
        lc.gather_local_stats(X, y)
    )

    augmented_X, augmented_X_labels = lc.add_site_covariates(args, X)
    augmented_X_labels = ["const"] + X_labels + augmented_X_labels

    beta_vec_size = augmented_X.shape[1]

    computation_output = {
        "output": {
            "beta_vec_size": beta_vec_size,
            "beta_vector_local": beta_vector,
            "number_of_regressions": len(y_labels),
            "computation_phase": "local_1",
            "augmented_X_labels": augmented_X_labels,
            "X_labels": X_labels
        },
        "cache": {
            "beta_vec_size": beta_vec_size,
            "number_of_regressions": len(y_labels),
            "covariates": augmented_X.tolist(),
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
    cost = np.zeros(number_of_regressions)

    for i in range(number_of_regressions):
        y_ = y[i]
        w_ = w[i]
        if not mask_flag[i]:
            gradient[i, :] = (
                1 / len(X)) * np.dot(biased_X.T, np.dot(biased_X, w_) - y_)
        cost[i] = lc.get_cost(y_actual=y[i], y_predicted=np.dot(biased_X, w_))

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
            "local_cost": cost.tolist(),
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
            sum_squared_error(curr_y, y_estimate(biased_X, avg_beta_vector)[index])
        )
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

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

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
