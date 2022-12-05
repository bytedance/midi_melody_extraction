# Copyright 2022 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT 
# 
# @Author: Katerina Kosta
# @Date:   2021-05-21 11:48:11
# @Last Modified by:   Katerina Kosta
# @Last Modified time: 2021-05-27 10:01:29

import argparse
import yaml
import logging
import os
import glob
import numpy as np
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stats_dict(data, output_folder):
    # Currently the features that are extracted are saved as stacked arrays.
    # The sequence of them is therefore hardcoded and setup in process_data.py
    # The outcome of the stats dictionary is a list, each element of which
    # corresponds to a specific feature, in the order of data_dict lists.

    stats_dict = {"min": [], "max": [], "mean": [], "std": []}

    no_of_features = data[0].shape[0]

    for i in range(no_of_features - 1):  # esclude last feature of the prediction
        flat_feature_elements = [item for sublist in [list(d[i]) for d in data] for item in sublist]
        stats_dict["min"].append(np.min(flat_feature_elements))
        stats_dict["max"].append(np.max(flat_feature_elements))
        stats_dict["mean"].append(np.mean(flat_feature_elements))
        stats_dict["std"].append(np.std(flat_feature_elements))

    with open(os.path.join(output_folder, "stats_config_train_valid.yaml"), "w") as f:
        yaml.dump(stats_dict, f)
        logger.info(f"stats_config.yaml metadata file created!")
    return


def main(opts):
    os.makedirs(opts.output, exist_ok=True)

    data_files = glob.glob(os.path.join(opts.input, "*.npz"))
    logger.info(f"Processing files in {opts.input} and saving the features in {opts.output}")

    random.shuffle(data_files)
    data_split = {}
    number_20 = (20 * len(data_files)) // 100

    data_split = {
        "test_files": data_files[:number_20],
        "valid_files": data_files[number_20 : 2 * number_20],
        "train_files": data_files[2 * number_20 :],
    }

    data = []
    for filename in data_split["train_files"] + data_split["valid_files"]:
        logger.info(f"Processing file {filename}")
        data.append(dict(np.load(filename, allow_pickle=True))["arr_0"])
    create_stats_dict(data, opts.output)
    with open(os.path.join(opts.output, "data_split.yaml"), "w") as f:
        yaml.dump(data_split, f)
        logger.info(f"data_split.yaml file created!")
    return


if __name__ == "__main__":
    # parse arguments from terminal
    parser = argparse.ArgumentParser(
        description="Process POP909 MIDI files to prepare them for melody extraction model"
    )
    parser.add_argument(
        "-i",
        action="store",
        dest="input",
        default="",
        help="Path to the .npz features folder",
    )

    parser.add_argument(
        "-o",
        action="store",
        dest="output",
        default="",
        help="Output directory for stats dictionary output file.",
    )

    opts = parser.parse_args()
    main(opts)
