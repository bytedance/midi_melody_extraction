# Copyright 2022 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT 
# 
# @Author: Katerina Kosta, Andrew Shaw
# @Date:   2021-05-21 11:48:11
# @Last Modified by:   Katerina Kosta
# @Last Modified time: 2022-08-29 14:20:10

import argparse
import logging
import os
import numpy as np
from data_tools import (
    read_MIDI_file,
    align_and_quantise_notes,
    compute_pitch,
    compute_dur,
    compute_pitch_dist,
    compute_pos_in_bar,
    compute_in_scale,
    compute_is_melody_note,
    get_key_signature,
    KeySignature,
    TimeSignature,
    maybe_raise_exception_for_time_signature,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_features(notes, key_signature, time_signature, include_melody_note_tag=True):
    # pitch
    pitches = compute_pitch(notes)
    # dur
    durations = compute_dur(notes)
    # pitch_dist_below
    pitch_dist_below = compute_pitch_dist(notes, dist_type="below")
    # pitch_dist_above
    pitch_dist_above = compute_pitch_dist(notes, dist_type="above")
    # pos_in_bar
    positions_in_bar = compute_pos_in_bar(notes, time_signature)
    # in_scale
    pitches_in_scale = compute_in_scale(notes, key_signature)

    if include_melody_note_tag:
        is_melody_note = compute_is_melody_note(notes)
        return (
            np.asarray(
                np.stack(
                    (
                        pitches,
                        durations,
                        pitch_dist_below,
                        pitch_dist_above,
                        positions_in_bar,
                        pitches_in_scale,
                    )
                ),
                dtype=np.float32,
            ),
            is_melody_note,
        )
    else:
        return np.asarray(
            np.stack(
                (
                    pitches,
                    durations,
                    pitch_dist_below,
                    pitch_dist_above,
                    positions_in_bar,
                    pitches_in_scale,
                )
            ),
            dtype=np.float32,
        )


def main(opts):
    POP909_path = opts.input
    output_folder = opts.output
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Processing files in {POP909_path} and saving the features in {opts.output}")

    for entry in [f for f in os.scandir(POP909_path) if not f.name.startswith(".")]:
        fpath = os.path.join(entry.path, entry.name + ".mid")
        logger.info(f"Processing file: {fpath}")
        # read the score
        notes = read_MIDI_file(fpath)

        if opts.preprocess:
            notes = align_and_quantise_notes(notes)
            # TODO: Plug here automatic downbeat alignment and other pre-processing steps.

        key_sig = get_key_signature(fpath)
        key_sig_event = KeySignature(name=key_sig)

        numerator, is_exception = maybe_raise_exception_for_time_signature(fpath)
        if is_exception:
            logger.info(f"Time signature numerator from data: {numerator}. Tagging it as '3/4'.")
            time_sig_event = TimeSignature(name="3/4")
        else:
            time_sig_event = TimeSignature(name="4/4")

        # compute the features
        features, target_predictions = compute_features(
            notes, key_sig_event, time_sig_event, include_melody_note_tag=True
        )
        finalised_features = np.vstack((features, target_predictions))

        # Save features
        entry_basename = os.path.basename(entry.name).split(".")[0]
        output_name = f"POP909-{entry_basename}_features"
        output_full_path = os.path.join(opts.output, output_name + ".npz")
        np.savez(output_full_path, finalised_features)
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
        default="_path_to_/POP909-Dataset/POP909 - MIDI folders",
        help="Path to the folder of POP909 dataset",
    )
    parser.add_argument(
        "-o",
        action="store",
        dest="output",
        default="",
        help="Output directory for output files that store the computed MIDI features.",
    )
    parser.add_argument(
        "-q",
        action="store_true",
        dest="preprocess",
        help="Add when we want to preprocess the notes of the input scores (quantisation).",
    )
    opts = parser.parse_args()

    main(opts)
