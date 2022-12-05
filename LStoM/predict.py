# Copyright 2022 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT 
# 
# @Author: Katerina Kosta
# @Date:   2021-05-21 11:48:11
# @Last Modified by:   Katerina Kosta
# @Last Modified time: 2021-05-27 10:47:41

import argparse
import os
import sys
import glob

import numpy as np
import torch

from LStoM.train import scale_or_normalise_data
from LStoM.data_tools import (
    read_MIDI_file,
    write_MIDI_melody_file,
    align_and_quantise_notes,
    infer_key_sig,
    KeySignature,
    TimeSignature,
)
from LStoM.process_data import compute_features

import sys


def get_predicted_labels(input_features, model_path):
    path2model = {}
    if model_path not in path2model:
        path2model[model_path] = torch.load(model_path, map_location=torch.device("cpu")).eval()
    model = path2model[model_path]

    input_features = torch.tensor(input_features.astype("float32"))
    input_features = torch.unsqueeze(input_features, dim=1)
    input_features = input_features.permute([2, 1, 0])  # [l, batch_size, no. of features]

    with torch.no_grad():
        pred = model(input_features)

    predicted_factors = np.squeeze(pred.numpy())
    return predicted_factors


def predict_from_file(fpath, model_path, stats_config_file, output_path, preprocess=False):
    os.makedirs(output_path, exist_ok=True)
    input_notes = read_MIDI_file(fpath)

    if preprocess:
        notes = align_and_quantise_notes(input_notes)

    key_sig_event = KeySignature(name=infer_key_sig(fpath))
    # TODO: Here to plug ways of detecting the MIDI time signature. Now it assumes 4/4.
    time_sig_event = TimeSignature(name="4/4")
    # compute the features
    features = compute_features(notes, key_sig_event, time_sig_event, include_melody_note_tag=False)

    scaled_features = scale_or_normalise_data(
        features.T, stats_config_file, "scale", data_includes_prediction=False
    )
    predictions = get_predicted_labels(scaled_features.T, model_path)
    mel_notes_loc = [bool(loc) for loc in np.round(predictions)]
    melody_notes = np.array(input_notes)[mel_notes_loc]
    output = os.path.join(output_path, os.path.basename(fpath).split(".")[0] + "_melody" + ".mid")
    write_MIDI_melody_file(melody_notes, output)
    return melody_notes


if __name__ == "__main__":
    # parse arguments from terminal
    parser = argparse.ArgumentParser(description="Predict melody from model")

    parser.add_argument(
        "-i", action="store", dest="input", default="", help="Path of the MIDI file."
    )

    parser.add_argument(
        "-sd",
        action="store",
        dest="stats_config_file",
        default="",
        help="Dictionary for features scaling factors.",
    )

    parser.add_argument("-m", action="store", dest="model_path", default="", help="Path of model.")

    parser.add_argument(
        "-o",
        action="store",
        dest="output_path",
        default="",
        help="Output folder path to save melody MIDI file.",
    )

    parser.add_argument(
        "-q",
        action="store_true",
        dest="preprocess",
        help="Add when we want to preprocess the notes of the input scores (quantisation).",
    )

    opts = parser.parse_args()
    predictions = predict_from_file(
        opts.input,
        opts.model_path,
        opts.stats_config_file,
        opts.output_path,
        opts.preprocess,
    )
