# Copyright 2022 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT 
# 
# @Author: Katerina Kosta
# @Date:   2021-05-21 11:48:11
# @Last Modified by:   Katerina Kosta
# @Last Modified time: 2022-08-29 10:02:37

import torch
from torch import nn


class LStoM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, bilstm=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.output_dim = 1
        self.num_directions = 2 if bilstm else 1
        self.lstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers, bidirectional=bilstm)
        self.fc = nn.Linear(self.hidden_size * self.num_directions, self.output_dim)

    def forward(self, x):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        seq_len, batch_size = x.shape[0], x.shape[1]

        h_t, c_t = (
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size),
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size),
        )

        outputs = []
        for i in x:
            input = i.view(1, -1, self.input_dim)
            out, (h_t, c_t) = self.lstm(input, (h_t.to(device), c_t.to(device)))
            out = self.fc(out)
            out = torch.squeeze(out, 1)
            out = torch.sigmoid(out)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=0)
        return outputs  # [seq_len, batch_size, 1]
