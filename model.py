import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # LSTM
        self.lstm_cell = nn.LSTMCell(1, 1)

        # decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

        # task specific MLPs
        self.contain = nn.Linear(1, 1)
        self.contact = nn.Linear(1, 1)
        self.stability = nn.Linear(1, 1)


    def forward(self, task, coordinates, images):
        if task == 'contact':
            self.linear = self.contact
        elif task == 'contain':
            self.linear = self.contain
        elif task == 'stability':
            self.linear = self.stability