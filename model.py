import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, first_n_frame_dynamics):
        super(Generator, self).__init__()

        # frame encoder
        self.conv1 = nn.Conv2d(3 * (first_n_frame_dynamics + 1), 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 8, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        # coordinate encoder
        self.coor = nn.Linear(6 * (first_n_frame_dynamics + 1), 64)

        # LSTM
        self.lstm_cell1 = nn.LSTMCell(8192 + 64, 2048 + 64)
        self.lstm_cell2 = nn.LSTMCell(2048 + 64, 2048 + 64)

        # coordinate decoder
        self.d_coor = nn.Linear(64, 6)

        # frame decoder
        self.upsample1 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.d_conv1 = nn.Conv2d(8, 32, 3, padding=1)
        self.d_conv2 = nn.Conv2d(32 * 2, 16, 3, padding=1)
        self.d_conv3 = nn.Conv2d(16 * 2, 3, 3, padding=1)

        # shared FC layer
        self.fc = nn.Linear(2048 + 64, 128)

        # task specific FC layers
        self.contain = nn.Linear(128, 1)
        self.contact = nn.Linear(128, 1)
        self.stability = nn.Linear(128, 1)


    def forward(self, task, coordinates, images, teacher_forcing_batch, first_n_frame_dynamics, device):
        sequence_len = len(images)
        batch_size, channels, height, width = images[0].shape
        decoded_images = []
        pred_coordinates = []
        assert first_n_frame_dynamics < sequence_len

        # stack first n frames into input for model to learn initial dynamics
        first_n_images_i = torch.zeros(batch_size, channels * (first_n_frame_dynamics + 1), height, width)
        first_n_coordinates_i = torch.zeros(batch_size, 6 * (first_n_frame_dynamics + 1)).float()
        for i in range(first_n_frame_dynamics):
            first_n_images_i[:,i*channels:(i+1)*channels,:,:] = images[i]
            first_n_coordinates_i[:,i*6:(i+1)*6] = torch.stack(coordinates[i]).T

        for i in range(first_n_frame_dynamics, sequence_len):
            images_i = first_n_images_i.clone()
            coordinates_i = first_n_coordinates_i.clone()
            if i == first_n_frame_dynamics:
                images_i[:,first_n_frame_dynamics*channels:,:,:] = images[i]
                coordinates_i[:,first_n_frame_dynamics*6:] = torch.stack(coordinates[i]).T
            elif i > first_n_frame_dynamics:
                for j in range(batch_size):
                    # teacher forcing
                    if teacher_forcing_batch[j]:
                        images_i[j,first_n_frame_dynamics*channels:,:,:] = images[i][j]
                        coordinates_i[j,first_n_frame_dynamics*6:] = torch.stack(coordinates[i]).T[j]
                    # no teacher forcing, use outputs from previous timestep
                    else:
                        images_i[j,first_n_frame_dynamics*channels:,:,:] = decoded_frame[j]
                        coordinates_i[j,first_n_frame_dynamics*6:] = decoded_coordinates[j]
            images_i = images_i.to(device)
            coordinates_i = coordinates_i.to(device)

            # encode frames
            conv1_feat = torch.relu(self.conv1(images_i))
            conv2_feat = torch.relu(self.conv2(conv1_feat))
            conv3_feat = torch.relu(self.conv3(conv2_feat))
            conv_flattened = self.flatten(conv3_feat)

            # encode coordinates input for object identification/tracking
            coor_feat = torch.relu(self.coor(coordinates_i))

            # merge encoded features
            concat_feat = torch.cat([conv_flattened, coor_feat], dim=1)

            # LSTM processing
            if i == first_n_frame_dynamics:
                hx1, cx1 = self.lstm_cell1(concat_feat)
                hx2, cx2 = self.lstm_cell2(hx1)
            else:
                hx1, cx1 = self.lstm_cell1(concat_feat, (hx1, cx1))
                hx2, cx2 = self.lstm_cell2(hx1, (hx2, cx2))

            # decode next coordinate predictions
            decoded_coordinates = hx2[:,2048:]
            decoded_coordinates = torch.relu(self.d_coor(decoded_coordinates))
            pred_coordinates.append(decoded_coordinates)

            # decode next frame predictions
            decoded_frame = torch.reshape(hx2[:,:2048], (batch_size, 8, 16, 16))
            decoded_frame = torch.relu(self.d_conv1(self.upsample1(decoded_frame)))
            decoded_frame = torch.relu(self.d_conv2(self.upsample2(torch.cat([decoded_frame, conv2_feat], dim=1))))
            decoded_frame = torch.relu(self.d_conv3(self.upsample2(torch.cat([decoded_frame, conv1_feat], dim=1))))
            decoded_images.append(decoded_frame)
        
        x = torch.relu(self.fc(hx2))
        if task == 'contact':
            out = self.contact(x)
        elif task == 'contain':
            out = self.contain(x)
        elif task == 'stability':
            out = self.stability(x)

        return out, decoded_images, pred_coordinates
