# Copyright 2022 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT 
# 
# @Author: Katerina Kosta
# @Date:   2021-05-21 11:48:11
# @Last Modified by:   Katerina Kosta
# @Last Modified time: 2021-05-27 10:13:38

import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import mir_eval
import yaml
import json

from LStoM.models import LStoM

import matplotlib.pyplot as plt
from datetime import datetime


logger = logging.getLogger()


def slice_data(source, step):
    segm = []
    for i in range(0, source.shape[1], step):
        segm.append(source[:, i : i + step])
    return segm


def make_segments(data_files, segment_size):
    all_segments = []
    for filename in data_files:
        data_dict = dict(np.load(filename, allow_pickle=True))
        data = data_dict["arr_0"]
        modulo = data.shape[1] % segment_size
        data = data[:, :-modulo]  # TODO: include the last segments as well doing padding
        segmented_data = slice_data(data, segment_size)

        all_segments.append(segmented_data)
    all_segments_flatten = [item for sublist in all_segments for item in sublist]
    return all_segments_flatten


def scale_data(data, stats_config_file, data_includes_prediction=True):
    with open(stats_config_file) as stats_file:
        stats_dict = yaml.load(stats_file, Loader=yaml.Loader)

    feature_range = len(data[0]) - 1  # exclude the last feature which are the predictions

    data_scaled = []
    for d in data:
        d_scaled = []
        for i in np.arange(feature_range):
            d_scaled.append((d[i] - stats_dict["mean"][i]) / (stats_dict["std"][i]))

        if data_includes_prediction:
            d_scaled.append(d[-1])

        data_scaled.append(np.array(d_scaled, dtype=np.float32))
    return np.array(data_scaled), feature_range


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    # overall accuracy
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / (y_test.shape[0] * y_test.shape[1])

    # accuracy in melody notes
    preds = y_pred_tag.detach().cpu().numpy()
    labels = y_test.detach().cpu().numpy()

    # MELODY PRECISION: MPR = number of correctly predicted melody notes / number of melody notes
    # MELODY RECALL: MR = number of correctly predicted melody notes / number of notes identified as melody
    # MELODY F measure: MF =  2 * ((MPR * MR) / (MPR + MR))
    number_of_correct_mn = 0
    for pred_list, label_list in list(zip(preds, labels)):
        for pr, l in list(zip(pred_list, label_list)):
            if l == 1 and pr == 1:
                number_of_correct_mn += 1

    MPR = number_of_correct_mn / np.sum(labels)
    MR = number_of_correct_mn / np.sum(preds)
    MF = 2 * ((MPR * MR) / (MPR + MR))

    number_of_wrong_mn = 0  # number of notes that are predicted as melody notes, although they are not. (False positive)
    for pred_list, label_list in list(zip(preds, labels)):
        for pr, l in list(zip(pred_list, label_list)):
            if l == 0 and pr == 1:
                number_of_wrong_mn += 1
    FA_manual = number_of_wrong_mn / (y_test.shape[0] * y_test.shape[1] - np.sum(labels))

    vx_R, vx_FA = mir_eval.melody.voicing_measures(labels, preds)

    return acc.detach().cpu().numpy(), MF, MPR, MR, FA_manual, vx_FA, vx_R


class WeightedFocalLoss(nn.Module):
    """
    Source: https://amaarora.github.io/2020/06/29/FocalLoss.html . Thanks!
    """

    def __init__(self, alpha=0.25, gamma=2):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        loss_calc = nn.BCELoss()
        BCE_loss = loss_calc(inputs, targets).to(device)
        targets = targets.type(torch.long).to(device)
        at = self.alpha.gather(0, targets.data.to(device).view(-1)).to(device)
        pt = torch.exp(-BCE_loss).to(device)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def main(output_path, data_split_file, stats_config_file, verbose, output_filename):
    now = datetime.now()
    logger.setLevel(logging.INFO) if verbose else logger.setLevel(logging.WARN)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    os.makedirs(output_path, exist_ok=True)

    with open(data_split_file) as data_split:
        data_split_dict = yaml.load(data_split, Loader=yaml.Loader)

    logger.info(f"Loading and segmenting data...")
    segmented_training_data = make_segments(data_split_dict["train_files"], 50)
    segmented_valid_data = make_segments(data_split_dict["valid_files"], 50)

    logger.info(f"Processing data...")
    processed_training_data, feature_range = scale_data(
        segmented_training_data, stats_config_file, data_includes_prediction=True
    )
    processed_valid_data, _ = scale_data(
        segmented_valid_data, stats_config_file, data_includes_prediction=True
    )

    logger.info(f"Starting training...")
    training_data = DataLoader(processed_training_data, batch_size=150, shuffle=True)
    valid_data = DataLoader(processed_valid_data, batch_size=150, shuffle=True)

    input_dim = feature_range  # how many features we feed into the network

    model = LStoM(input_dim, hidden_size=140, num_layers=6, bilstm=True)
    optimizer = torch.optim.Adam(model.parameters())

    criterion = WeightedFocalLoss()

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    train_loss_list = []
    valid_loss_list = []
    eval_every = 52
    current_accuracy = 0.0
    current_MF = 0
    current_MPR = 0
    current_MR = 0
    current_vx_FA = 0
    current_vx_R = 0

    accuracy = []
    mel_MF = []
    mel_MPR = []
    mel_MR = []
    mel_vx_FA = []
    mel_vx_R = []
    patience = 0

    model.train().to(device)

    for epoch in range(150):
        if patience < 3:

            # Training step
            for data in training_data:
                data = data.to(device)
                feat = data[:, :-1].to(device)
                labels = data[:, -1].to(device)
                optimizer.zero_grad()
                input_features = feat.permute([2, 0, 1])  # I want: [l, batch_size, no. of features]
                preds = model(input_features)
                preds = preds.squeeze().T
                labels = labels.squeeze()
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Evaluation step
            model.eval().to(device)
            with torch.no_grad():
                for data in valid_data:
                    feat = data[:, :-1].to(device)
                    labels = data[:, -1].to(device)
                    input_features = feat.permute([2, 0, 1])

                    results = model(input_features).to(device)
                    preds = results.squeeze().T
                    preds = preds.squeeze()
                    loss = criterion(preds, labels)
                    valid_running_loss += loss.item()
                    acc, MF, MPR, MR, FA_manual, vx_FA, vx_R = binary_acc(preds, labels)
                    current_accuracy += acc
                    current_MF += MF
                    current_MPR += MPR
                    current_MR += MR
                    current_vx_FA += vx_FA
                    current_vx_R += vx_R

            average_train_loss = running_loss / eval_every
            average_valid_loss = valid_running_loss / len(valid_data)
            train_loss_list.append(average_train_loss)
            valid_loss_list.append(average_valid_loss)

            average_accuracy = current_accuracy / len(valid_data)
            average_mel_MF = np.round(current_MF / len(valid_data), 3)
            average_mel_MPR = np.round(current_MPR / len(valid_data), 3)
            average_mel_MR = np.round(current_MR / len(valid_data), 3)
            average_mel_vx_FA = np.round(current_vx_FA / len(valid_data), 3)
            average_mel_vx_R = np.round(current_vx_R / len(valid_data), 3)

            accuracy.append(average_accuracy)
            mel_MF.append(average_mel_MF)
            mel_MPR.append(average_mel_MPR)
            mel_MR.append(average_mel_MR)
            mel_vx_FA.append(average_mel_vx_FA)
            mel_vx_R.append(average_mel_vx_R)

            if average_mel_MF < np.mean(mel_MF[-5:]):
                patience += 1
            else:
                patience = 0

            # resetting running values
            running_loss = 0.0
            valid_running_loss = 0.0
            current_accuracy = 0.0
            current_MF = 0
            current_MPR = 0
            current_MR = 0
            current_vx_FA = 0
            current_vx_R = 0
            model.train()

            logger.info(
                f"Epoch: {epoch}, Train loss: {average_train_loss}, Validation loss: {average_valid_loss}, Accuracy: {average_accuracy}, Melody F measure: {average_mel_MF}, PR R mirR: {average_mel_MPR, average_mel_MR, average_mel_vx_R}, FA: {average_mel_vx_FA}, Patience: {patience}"
            )

        else:
            break

    logger.info("\nTraining finished!")

    if not output_filename:
        output_filename = "epoch" + str(epoch)
    model_name = now.strftime(output_filename + "_model.pt")
    model_path = os.path.join(output_path, model_name)
    torch.save(model, model_path)

    logger.info("Creating loss graph...")
    plt.figure()
    plt.plot(train_loss_list)
    plt.ylabel("loss")
    plt.xlabel("step")
    plt.savefig(os.path.join(output_path, output_filename + "losses.png"))

    logger.info("Creating melody F measure graph...")
    plt.figure()
    plt.plot(mel_MF)
    plt.ylabel("Melody notes F measure")
    plt.xlabel("step")
    plt.savefig(os.path.join(output_path, output_filename + "mel_F.png"))

    outputs = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "average_accuracy": average_accuracy,
        "mel_MF": mel_MF,
        "mel_MPR": mel_MPR,
        "mel_MR": mel_MR,
        "mel_vx_FA": mel_vx_FA,
        "mel_vx_R": mel_vx_R,
    }

    with open(os.path.join(output_path, output_filename + "_output.json"), "w") as f:
        json.dump(outputs, f)
    return


if __name__ == "__main__":
    # parse arguments from terminal
    parser = argparse.ArgumentParser(description="Train model for melody extraction task")

    parser.add_argument(
        "-o",
        action="store",
        dest="output",
        default="",
        help="Directory for output files",
    )

    parser.add_argument("-v", action="store_true", dest="verbose", help="Set verbose mode.")

    parser.add_argument(
        "-sd",
        action="store",
        dest="stats_config_file",
        default="",
        help="Dictionary for features scaling factors",
    )

    parser.add_argument(
        "-ds",
        action="store",
        dest="data_split_file",
        default="",
        help="Dictionary for data train/valid/test split",
    )

    parser.add_argument(
        "-of",
        action="store",
        dest="output_filename",
        help="filename for the saved model. If no string is provided, the filename will be the [current_time]_[epoch]_model.pt",
        default="",
    )

    opts = parser.parse_args()

    main(
        opts.output,
        opts.data_split_file,
        opts.stats_config_file,
        opts.verbose,
        opts.output_filename,
    )
