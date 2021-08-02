#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from utils import utils
from utils.readers import InHospitalMortalityReader
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import keras_utils
from utils import common_utils
from utils.arg_metav_formatter import *
from keras_models.lstm import *
from keras.callbacks import ModelCheckpoint, CSVLogger


def initialize_var():
    """
    Function to initialize variables and update them in global namespace;
    default values adopted from MIMIC-III benchmarks github readme
    and best model runs
    """
    data_dir = "./data/in-hospital-mortality"
    load_state = ("./keras_models/best_models/ihm/" +
                  "rk_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch27." +
                  "test0.278806287862.state")
    load_model = "./keras_models/lstm.py"
    output_dir = "./output"
    target_repl_coef = 0.0
    timestep = 1.0
    imputation = "previous"
    small_part = False
    batch_size = 8
    mode = "test"
    optimizer = "adam"
    lr = 0.001
    beta_1 = 0.9
    target_repl = (target_repl_coef > 0.0 and mode == 'train')
    normalizer_state = ("./utils/resources/ihm_ts1.0.input_str" +
                        ":previous.start_time:zero.normalizer")
    # NOTE updating globals preferred over argparse to allow for both
    # command-line execution and interactive debugging
    globals().update(locals())


def load_test_data_ihm():
    """
    Function to initialize MIMIC-III benchmark IHM test-data

    Returns:
        data (numpy.ndarray): discretized and normalized IHM data
        labels (list[int]): true values of IHM task
        names (list[str]): episode reference for true IHM value
        discretizer_header (list[str]): header information on each variable
    """
    discretizer = Discretizer(timestep=float(timestep),
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')
    # Build readers, discretizers, normalizers
    train_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(data_dir, 'train'),
        listfile=os.path.join(data_dir, 'train_listfile.csv'),
        period_length=48.0)
    discretizer_header = discretizer.transform(
        train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [
        i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
    ]
    normalizer = Normalizer(fields=cont_channels)
    normalizer.load_params(normalizer_state)
    test_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(data_dir, 'test'),
        listfile=os.path.join(data_dir, 'test_listfile.csv'),
        period_length=48.0)
    ret = utils.load_data(test_reader,
                          discretizer,
                          normalizer,
                          small_part,
                          return_names=True)
    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]
    return data, labels, names, discretizer_header


def load_best_model_ihm():
    """
    Function to return best model for MIMIC-III benchmark IHM task

    Returns:
        model (Network): custom Network class as defined in
        './keras_models/lstm.py'
    """
    # create arg dict
    args_dict = {}
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl
    args_dict["dim"] = 16
    args_dict["batch_norm"] = False
    args_dict["dropout"] = 0.3
    args_dict["rec_dropout"] = 0.0
    args_dict["depth"] = 2
    # build the model
    model = Network(**args_dict)
    # create optimizer
    optimizer_config = {
        'class_name': optimizer,
        'config': {
            'lr': lr,
            'beta_1': beta_1
        }
    }
    # compute loss style
    if target_repl:
        loss = ['binary_crossentropy'] * 2
        loss_weights = [1 - target_repl_coef, target_repl_coef]
    else:
        loss = 'binary_crossentropy'
        loss_weights = None
    # compile model
    model.compile(optimizer=optimizer_config,
                  loss=loss,
                  loss_weights=loss_weights)
    # load model weights
    model.load_weights(load_state)
    return model


def create_feature_indices(header):
    """
    Function to return unique features along with respective column indices
    for each feature in the final numpy array

    Args:
        header (list[str]): description of each feature's possible values

    Returns:
        feature_indices (dict): unique feature names as keys with value
        types (dicrete or continuous) and data column indices where present
    """
    feature_indices = {}
    for i, head in enumerate(header):
        current = head.split("->")
        str_name = current[0].replace(" ", "_")
        if current[0] == "mask":
            feature_indices["presence_" +
                            current[1].replace(" ", "_")] = ["discrete", i]
        elif feature_indices == {} or str_name not in feature_indices:
            if len(current) > 1:
                feature_indices[str_name] = ["discrete", i]
            else:
                feature_indices[str_name] = ["continuous", i]
        elif str_name in feature_indices:
            feature_indices[str_name].extend([i])
    return feature_indices


def zero_normal_occlude(feature_indices, data, model, occlusion_type, pd_df,
                        header):
    """
    Function to conduct zero or normal-value occlusion. Zero occlusion means
    replacing the feature with zeroes. Given the best model for the IHM task,
    this would imply that the final model would mask out the zero-values.

    Normal-occlusion refers to replacing the feature of interest with the
    "normal" or "healthy" values present in 'dicretizer_config.json'. For
    discrete features, this would equate to a one-hot encoding of the "normal"
    categorical feature. For continuous features, this would equate to replacing
    the value with the normalized "normal" value.

    Args:
        feature_indices (dict): unique feature names as keys with value
        types (dicrete or continuous) and data column indices where present
        data (numpy.ndarray): normalized and discretized dataset
        model (Network): model of custom Network class
        occlusion_type (str): type of occlusion to conduct, either 'zero' or
        'normal-value'
        pd_df (pandas.core.frame.DataFrame): existing pandas DataFrame to append
        results onto
        header (list[str]): description of each feature's possible values

    Returns:
        pd_df (pandas.core.frame.DataFrame): modified pandas DataFrame with
        results from occlusion
    """
    if occlusion_type == "zero":
        for key in feature_indices.keys():
            cols = feature_indices[key][1:]
            occluded_data = data.copy()
            occluded_data[:, :, cols] = 0
            predictions = model.predict(occluded_data,
                                        batch_size=batch_size,
                                        verbose=1)
            predictions = np.array(predictions)[:, 0]
            pd_df[key] = predictions
    elif occlusion_type == "normal-value":
        with open("./utils/resources/discretizer_config.json", "r") as f:
            config = json.load(f)
            possible_values = config['possible_values']
            normal_values = config['normal_values']
        normalizer = Normalizer()
        normalizer.load_params(normalizer_state)
        for key in feature_indices.keys():
            full_name = key.replace("_", " ")
            if feature_indices[key][0] == "discrete":
                if "presence_" in key:
                    to_sub = 1
                else:
                    normal_value = normal_values[full_name]
                    category_id = possible_values[full_name].index(
                        normal_value)
                    N_values = len(possible_values[full_name])
                    one_hot = np.zeros((N_values, ))
                    one_hot[category_id] = 1
                    to_sub = one_hot
            elif feature_indices[key][0] == "continuous":
                index = header.index(full_name)
                to_sub = ((float(normal_values[full_name]) -
                           normalizer._means[index]) / normalizer._stds[index])
            cols = feature_indices[key][1:]
            occluded_data = data.copy()
            occluded_data[:, :, cols] = to_sub
            predictions = model.predict(occluded_data,
                                        batch_size=batch_size,
                                        verbose=1)
            predictions = np.array(predictions)[:, 0]
            pd_df[key] = predictions
    if not os.path.isdir(os.path.join("./output/", occlusion_type)):
        os.mkdir(os.path.join("./output/", occlusion_type))
    pd_df.to_csv(os.path.join("./output/", occlusion_type, "result.csv"),
                 index=False)
    return pd_df


def inner_outer_occlude(feature_indices, data, model, n_iterations,
                        occlusion_type, pd_df):
    """
    Function to conduct inner or outer occlusion. Inner occlusion refers
    occluding individual data instances by shuffling the feature values within
    the individual dataset. By doing this, any correlations that may exist on a
    single feature would be perturbed. This permutation can be repeated multiple
    times to measure significant statistical effect.

    Outer occlusion refers to shuffling a feature within all data instances
    instead of within a single data instance. This would also disrupt any
    potential correlations between the feature of interest, but the
    perturbation would be slightly greater since the respective features
    of all samples can be inter-swapped. This permutation can also be repeated
    multiple times to measure significant statistical effects.

    Args:
        feature_indices (dict): unique feature names as keys with value
        types (dicrete or continuous) and data column indices where present
        data (numpy.ndarray): normalized and discretized dataset
        model (Network): model of custom Network class
        occlusion_type (str): type of occlusion to conduct, either 'inner' or
        'outer'
        n_iterations (int): number of times to permute data in occlusion
        pd_df (pandas.core.frame.DataFrame): existing pandas DataFrame to append
        results onto

    Returns:
        pd_df (pandas.core.frame.DataFrame): modified pandas DataFrame with
        results from occlusion
    """
    if occlusion_type == "inner":
        for key in feature_indices.keys():
            cols = feature_indices[key][1:]
            for i in range(n_iterations):
                shuffled_data = data.copy()
                indices = list(range(data.shape[1]))
                np.random.shuffle(indices)
                for j in range(shuffled_data.shape[0]):
                    for k, l in enumerate(indices):
                        shuffled_data[j][k, cols] = data[j][l, cols].copy()
                predictions = model.predict(shuffled_data,
                                            batch_size=batch_size,
                                            verbose=1)
                predictions = np.array(predictions)[:, 0]
                pd_df[key + "_" + str(i)] = predictions
    elif occlusion_type == "outer":
        for key in feature_indices.keys():
            cols = feature_indices[key][1:]
            for i in range(n_iterations):
                shuffled_data = data.copy()
                indices = list(range(data.shape[0]))
                np.random.shuffle(indices)
                for j, k in enumerate(indices):
                    shuffled_data[j][:, cols] = data[k][:, cols].copy()
                predictions = model.predict(shuffled_data,
                                            batch_size=batch_size,
                                            verbose=1)
                predictions = np.array(predictions)[:, 0]
                pd_df[key + "_" + str(i)] = predictions
    if not os.path.isdir(os.path.join("./output/", occlusion_type)):
        os.mkdir(os.path.join("./output/", occlusion_type))
    pd_df.to_csv(os.path.join("./output/", occlusion_type, "result.csv"),
                 index=False)
    return pd_df


def occlude(occlusion_type="all", n_iterations=5):
    """
    Main function to conduct occlusion

    Args:
        occlusion_type (str): type of occlusion to conduct, either 'zero',
        'normal-value', 'inner' or 'outer'
        n_iterations (int): number of times to permute data in occlusion

    Returns:
        pd_df (pandas.core.frame.DataFrame|list[pandas.core.frame.DataFrame]):
        if occlusion_type == 'all'; then list of pandas DataFrames are returned.
        Otherwise, modified pandas DataFrame with occlusion results is returned
    """
    initialize_var()
    data, labels, names, header = load_test_data_ihm()
    model = load_best_model_ihm()
    # get important feature indices
    feature_indices = create_feature_indices(header)
    # run true results
    base_predictions = model.predict(data, batch_size=batch_size, verbose=1)
    base_predictions = np.array(base_predictions)[:, 0]
    res_dict = {
        "instance": names,
        "true": labels,
        "best_score": base_predictions
    }
    pd_df = pd.DataFrame(res_dict, columns=["instance", "true", "best_score"])
    # run occlusion functions
    if occlusion_type != "all":
        if occlusion_type in ["zero", "normal-value"]:
            pd_df = zero_normal_occlude(feature_indices, data, model,
                                        occlusion_type, pd_df, header)
        elif occlusion_type in ["outer", "inner"]:
            pd_df = inner_outer_occlude(feature_indices, data, model,
                                        n_iterations, occlusion_type, pd_df)
    else:
        occlusion_type = "zero"
        pd_df_zero = zero_normal_occlude(feature_indices, data,
                                         model, occlusion_type,
                                         pd_df.copy(deep=True), header)
        occlusion_type = "normal-value"
        pd_df_nv = zero_normal_occlude(feature_indices, data,
                                       model, occlusion_type,
                                       pd_df.copy(deep=True), header)
        occlusion_type = "inner"
        pd_df_inner = inner_outer_occlude(feature_indices, data, model,
                                          n_iterations, occlusion_type,
                                          pd_df.copy(deep=True))
        occlusion_type = "outer"
        pd_df_outer = inner_outer_occlude(feature_indices, data, model,
                                          n_iterations, occlusion_type,
                                          pd_df.copy(deep=True))
        pd_df = [pd_df_zero, pd_df_nv, pd_df_inner, pd_df_outer]
    return pd_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--occlusion-type",
                        type=str,
                        default="all",
                        help="type of occlusion to conduct, either" +
                        " 'zero', 'normal-value', 'inner', 'outer'" +
                        " or 'all'")
    inner_outer = parser.add_argument_group(
        "arguments specific to inner/outer" + " occlusion")
    inner_outer.add_argument(
        "--n-iterations",
        type=int,
        default=5,
        help="number of permutation iterations for inner" + "/outer occlusion")
    args = parser.parse_args()
    occlude(args.occlusion_type, args.n_iterations)
