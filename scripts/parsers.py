#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 19:25:26 2018

@author: Harshvardhan
"""
import os
import pandas as pd
import nibabel as nib


def parse_for_each_y(args, X_files, y_files, dependent):
    y = []
    for file in y_files:
        if file.split('-')[-1] in X_files:
            with open(
                    os.path.join(args["state"]["baseDirectory"],
                                 file)) as fh:
                for line in fh:
                    if line.startswith(dependent[0]):
                        y.append(float(line.split('\t')[1]))

    return y


def parse_for_y_array(args, X_files, y_files, y_labels):
    y_array = []
    for region in y_labels:
        y_array.append(parse_for_each_y(args, X_files, y_files, region))

    y_array = list(map(list, zip(*y_array)))

    return y_array


def fsl_parser(args):
    input_list = args["input"]
    X_info = input_list["covariates"]
    y_info = input_list["data"]

    X_data = X_info[0][0]
    X_labels = X_info[1]
    X_types = X_info[2]

    X_df = pd.DataFrame.from_records(X_data)
    X_df.columns = X_df.iloc[0]
    X_df = X_df.reindex(X_df.index.drop(0))

    X_files = list(X_df['freesurferfile'])

    X = X_df[X_labels]
    xs = X_labels
    ys = X_types
    result = [x for x, y in zip(xs, ys) if y == 'boolean']
    pd.options.mode.chained_assignment = None  # default='warn'
    X[result] = (X[result] == 'True').astype(int)
    X = X.apply(pd.to_numeric, errors='ignore')

    y_files = y_info[0]
    y_labels = y_info[2]

    y_list = parse_for_y_array(args, X_files, y_files, y_labels)
    y = pd.DataFrame.from_records(y_list, columns=y_labels)

    return (
        X, y, y_labels
    )  # Ask about labels being available at this level without being passed around


def nifti_to_data(args, X_files, y_files):
    mask_file = os.path.join(args["state"]["baseDirectory"], 'mask_6mm.nii')
    mask_data = nib.load(mask_file).get_data()

    appended_data = []

    # Extract Data (after applying mask)
    for image in X_files:
        if image in y_files:
            image_data = nib.load(
                os.path.join(args["state"]["baseDirectory"],
                             image)).get_data()
            appended_data.append(image_data[mask_data > 0])

    return appended_data


def vbm_parser(args):
    input_list = args["input"]
    X_info = input_list["covariates"]
    y_info = input_list["data"]

    X_data = X_info[0][0]
    X_labels = X_info[1]
    #    X_types = X_info[2]

    X_df = pd.DataFrame.from_records(X_data)
    X_df.columns = X_df.iloc[0]
    X_df = X_df.reindex(X_df.index.drop(0))

    X_files = list(X_df['niftifile'])

    X = X_df[X_labels]
    X = X.apply(pd.to_numeric, errors='ignore')
    X = pd.get_dummies(X, drop_first=True)

    y_files = y_info[0]

    y_list = nifti_to_data(args, X_files, y_files)
    y = pd.DataFrame.from_records(y_list)

    X.isControl = X.isControl.astype(int)  # need to generalize this line

    return (X, y)
