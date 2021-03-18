import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # LSTM
        self.lstm_cell = nn.LSTMCell(5488, 256)

        # decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

        # task agnostic FC layer
        self.fc = nn.Linear(256 + 4, 64)

        # task specific FC layers
        self.contain = nn.Linear(64, 1)
        self.contact = nn.Linear(64, 1)
        self.stability = nn.Linear(64, 1)


    def forward(self, task, coordinates, images):
        if task == 'contact':
            self.task_fc = self.contact
        elif task == 'contain':
            self.task_fc = self.contain
        elif task == 'stability':
            self.task_fc = self.stability

        sequence_len = len(images)
        batch_size = images[0].shape[0]
        hx = torch.randn(batch_size, 256)
        cx = torch.randn(batch_size, 256)
        for i in range(sequence_len):
            # encoder frame features
            x = torch.relu(self.conv1(images[i]))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = self.flatten(x)

            # LSTM processing
            hx, cx = self.lstm_cell(x, (hx, cx))
        
        # add coordinate conditioning for object identification/tracking
        x = torch.cat([hx, coordinates.type_as(hx)], dim=1)
        x = self.fc(x)
        x = torch.relu(x)
        out = self.task_fc(x)

        return out