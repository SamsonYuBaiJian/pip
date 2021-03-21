import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 8, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        # LSTM
        self.lstm_cell1 = nn.LSTMCell(8192, 2048)
        self.lstm_cell2 = nn.LSTMCell(2048, 2048)

        # decoder
        self.upsample1 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.d_conv1 = nn.Conv2d(8, 32, 3, padding=1)
        self.d_conv2 = nn.Conv2d(32 * 2, 16, 3, padding=1)
        self.d_conv3 = nn.Conv2d(16 * 2, 3, 3, padding=1)

        # shared FC layer
        self.fc = nn.Linear(2048, 128)

        # task specific FC layers
        self.contain = nn.Linear(128, 1)
        self.contact = nn.Linear(128 + 4, 1)
        self.stability = nn.Linear(128, 1)


    def forward(self, task, coordinates, images, device):
        sequence_len = len(images)
        batch_size = images[0].shape[0]
        decoded_images = []

        for i in range(sequence_len):
            # encode frames
            image = images[i].to(device)
            conv1_feat = torch.relu(self.conv1(image))
            conv2_feat = torch.relu(self.conv2(conv1_feat))
            conv3_feat = torch.relu(self.conv3(conv2_feat))
            flattened = self.flatten(conv3_feat)

            # LSTM processing
            if i == 0:
                hx1, cx1 = self.lstm_cell1(flattened)
                hx2, cx2 = self.lstm_cell2(hx1)
            else:
                hx1, cx1 = self.lstm_cell1(flattened, (hx1, cx1))
                hx2, cx2 = self.lstm_cell2(hx1, (hx2, cx2))

            # decode next frame predictions
            decoded_x = torch.reshape(hx2, (batch_size, 8, 16, 16))
            decoded_x = torch.relu(self.d_conv1(self.upsample1(decoded_x)))
            decoded_x = torch.relu(self.d_conv2(self.upsample2(torch.cat([decoded_x, conv2_feat], dim=1))))
            decoded_x = torch.relu(self.d_conv3(self.upsample2(torch.cat([decoded_x, conv1_feat], dim=1))))
            decoded_images.append(decoded_x)
        
        x = torch.relu(self.fc(hx2))
        # add coordinate conditioning for object identification/tracking
        x = torch.cat([x, coordinates.type_as(x)], dim=1)
        if task == 'contact':
            out = self.contact(x)
        elif task == 'contain':
            out = self.contain(x)
        elif task == 'stability':
            out = self.stability(x)

        return out, decoded_images
