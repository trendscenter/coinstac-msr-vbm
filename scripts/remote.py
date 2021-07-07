#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for single-shot ridge
regression with decentralized statistic calculation
"""
import json
import sys
import scipy as sp
import numpy as np
import regression as reg


def remote_0(args):
    input_list = args["input"]
    site_ids = list(input_list.keys())
    site_covar_list = [
        '{}_{}'.format('site', label)
        for index, label in enumerate(site_ids) if index
    ]

    computation_output_dict = {
        "output": {
            "site_covar_list": site_covar_list,
            "computation_phase": "remote_0"
        },
        "cache": {}
    }

    return json.dumps(computation_output_dict)


def remote_1(args):
    """Need this function for performing multi-shot regression"""
    input_list = args["input"]
    first_user_id = list(input_list.keys())[0]
    beta_vec_size = input_list[first_user_id]["beta_vec_size"]
    number_of_regressions = input_list[first_user_id]["number_of_regressions"]

    # Initial setup
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    tol = 100  # 0.01
    eta = 1000  # 0.05
    count = 0

    wp = np.zeros((number_of_regressions, beta_vec_size), dtype=float)
    wc = np.zeros((number_of_regressions, beta_vec_size), dtype=float)
    mt = np.zeros((number_of_regressions, beta_vec_size), dtype=float)
    vt = np.zeros((number_of_regressions, beta_vec_size), dtype=float)

    iter_flag = 1

    computation_output = {
        "cache": {
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "tol": tol,
            "eta": eta,
            "count": count,
            "wp": wp.tolist(),
            "wc": wc.tolist(),
            "mt": mt.tolist(),
            "vt": vt.tolist(),
            "iter_flag": iter_flag,
            "number_of_regressions": number_of_regressions,
        },
        "output": {
            "remote_beta": wp.tolist(),
            "iter_flag": iter_flag,
            "computation_phase": "remote_1"
        }
    }

    return json.dumps(computation_output)


def remote_2(args):

    beta1 = args["cache"]["beta1"]
    beta2 = args["cache"]["beta2"]
    eps = args["cache"]["eps"]
    tol = args["cache"]["tol"]
    eta = args["cache"]["eta"]
    count = args["cache"]["count"]
    wp = np.array(args["cache"]["wp"], dtype=float)
    wc = np.array(args["cache"]["wc"], dtype=float)
    mt = args["cache"]["mt"]
    vt = args["cache"]["vt"]
    iter_flag = args["cache"]["iter_flag"]
    number_of_regressions = args["cache"]["number_of_regressions"]

    count = count + 1

    if not iter_flag:
        computation_output = {
            "cache": {
                "avg_beta_vector": wc.tolist()
            },
            "output": {
                "avg_beta_vector": wc.tolist(),
                "computation_phase": "remote_2b"
            }
        }
    else:
        input_list = args["input"]
        if len(input_list) == 1:
            grad_remote = [
                np.array(args["input"][site]["local_grad"])
                for site in input_list
            ]
            grad_remote = grad_remote[0]
        else:
            grad_remote = sum([
                np.array(args["input"][site]["local_grad"])
                for site in input_list
            ])

        mt = beta1 * np.array(mt) + (1 - beta1) * grad_remote
        vt = beta2 * np.array(vt) + (1 - beta2) * (grad_remote**2)

        m = mt / (1 - beta1**count)
        v = vt / (1 - beta2**count)

        wc = wp - eta * m / (np.sqrt(v) + eps)

        mask_flag = np.linalg.norm(wc - wp, axis=1) <= tol

        if sum(mask_flag) == number_of_regressions:
            iter_flag = 0

        for i in range(mask_flag.shape[0]):
            if not mask_flag[i]:
                wp[i] = wc[i]

        computation_output = {
            "cache": {
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps,
                "tol": tol,
                "eta": eta,
                "count": count,
                "wp": wp.tolist(),
                "wc": wc.tolist(),
                "mt": mt.tolist(),
                "vt": vt.tolist(),
                "iter_flag": iter_flag
            },
            "output": {
                "remote_beta": wc.tolist(),
                "mask_flag": mask_flag.astype(int).tolist(),
                "computation_phase": "remote_2a"
            }
        }

    return json.dumps(computation_output)


def remote_3(args):
    """Computes the global beta vector, mean_y_global & dof_global

    Args:
        args (dictionary): {"input": {
                                "beta_vector_local": list/array,
                                "mean_y_local": list/array,
                                "count_local": int,
                                "computation_phase": string
                                },
                            "cache": {}
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": list,
                                        "mean_y_global": ,
                                        "computation_phase":
                                        },
                                    "cache": {
                                        "avg_beta_vector": ,
                                        "mean_y_global": ,
                                        "dof_global":
                                        },
                                    }

    """
    input_list = args["input"]
    first_user_id = list(input_list)[0]

    avg_beta_vector = np.array(args["cache"]["avg_beta_vector"])

    all_local_stats_dicts = [
        input_list[site]["local_stats_list"] for site in input_list
    ]

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [
        np.array(input_list[site]["count_local"]) for site in input_list
    ]
    mean_y_global = np.array(mean_y_local) * np.array(count_y_local)
    mean_y_global = np.average(mean_y_global, axis=0)

    dof_global = sum(count_y_local) - avg_beta_vector.shape[1]

    computation_output = {
        "output": {
            "avg_beta_vector": avg_beta_vector.tolist(),
            "mean_y_global": mean_y_global.tolist(),
            "computation_phase": "remote_3"
        },
        "cache": {
            "avg_beta_vector": avg_beta_vector.tolist(),
            "mean_y_global": mean_y_global.tolist(),
            "dof_global": dof_global.tolist(),
            "all_local_stats_dicts": all_local_stats_dicts,
            "y_labels": args["input"][first_user_id]["y_labels"]
        },
    }

    return json.dumps(computation_output)


def remote_4(args):
    """
    Computes the global model fit statistics, r_2_global, ts_global, ps_global

    Args:
        args (dictionary): {"input": {
                                "SSE_local": ,
                                "SST_local": ,
                                "varX_matrix_local": ,
                                "computation_phase":
                                },
                            "cache":{},
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": ,
                                        "beta_vector_local": ,
                                        "r_2_global": ,
                                        "ts_global": ,
                                        "ps_global": ,
                                        "dof_global":
                                        },
                                    "success":
                                    }
    Comments:
        Generate the local fit statistics
            r^2 : goodness of fit/coefficient of determination
                    Given as 1 - (SSE/SST)
                    where   SSE = Sum Squared of Errors
                            SST = Total Sum of Squares
            t   : t-statistic is the coefficient divided by its standard error.
                    Given as beta/std.err(beta)
            p   : two-tailed p-value (The p-value is the probability of
                  seeing a result as extreme as the one you are
                  getting (a t value as large as yours)
                  in a collection of random data in which
                  the variable had no effect.)

    """
    input_list = args["input"]
    y_labels = args["cache"]["y_labels"]
    all_local_stats_dicts = args["cache"]["all_local_stats_dicts"]

    cache_list = args["cache"]
    avg_beta_vector = cache_list["avg_beta_vector"]
    dof_global = cache_list["dof_global"]

    SSE_global = sum(
        [np.array(input_list[site]["SSE_local"]) for site in input_list])
    SST_global = sum(
        [np.array(input_list[site]["SST_local"]) for site in input_list])
    varX_matrix_global = sum([
        np.array(input_list[site]["varX_matrix_local"]) for site in input_list
    ])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)

    ts_global = []
    ps_global = []

    for i in range(len(MSE)):
        var_covar_beta_global = MSE[i] * sp.linalg.inv(varX_matrix_global)
        se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
        ts = avg_beta_vector[i] / se_beta_global
        ps = reg.t_to_p(ts, dof_global[i])
        ts_global.append(ts)
        ps_global.append(ps)

    # Block of code to print local stats as well
    sites = ['Site_' + str(i) for i in range(len(all_local_stats_dicts))]

    all_local_stats_dicts = list(map(list, zip(*all_local_stats_dicts)))

    a_dict = [{
        key: value
        for key, value in zip(sites, all_local_stats_dicts[i])
    } for i in range(len(all_local_stats_dicts))]

    # Block of code to print just global stats
    keys1 = [
        "avg_beta_vector", "r2_global", "ts_global", "ps_global", "dof_global"
    ]
    global_dict_list = []
    for index, _ in enumerate(y_labels):
        values = [
            avg_beta_vector[index], r_squared_global[index],
            ts_global[index].tolist(), ps_global[index], dof_global[index]
        ]
        my_dict = {key: value for key, value in zip(keys1, values)}
        global_dict_list.append(my_dict)

    # Print Everything
    dict_list = []
    keys2 = ["ROI", "global_stats", "local_stats"]
    for index, label in enumerate(y_labels):
        values = [label, global_dict_list[index], a_dict[index]]
        my_dict = {key: value for key, value in zip(keys2, values)}
        dict_list.append(my_dict)

    computation_output = {
        "output": {
            "regressions": dict_list
        },
        "success": True
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(reg.listRecursive(parsed_args, "computation_phase"))

    if "local_0" in phase_key:
        computation_output = remote_0(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_2" in phase_key:
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_3" in phase_key:
        computation_output = remote_3(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_4" in phase_key:
        computation_output = remote_4(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
