import pandas as pd
from torch.utils.data import Dataset
import os
from skimage import io
from torchvision import transforms
from natsort import natsorted
import torch
import numpy as np


class Data(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = {'index': [], 'coordinates': [], 'labels': []}
        df = pd.read_csv(csv_file)
        for row in df.itertuples():
            # gets rid of NaN values
            row_data = [i for i in row if i == i]
            # each object has 6 data values
            index = int(row_data[0])
            row_data = row_data[1:]
            num_objects = int(len(row_data) / 6)
            for i in range(num_objects):
                self.data['index'].append(index)
                self.data['coordinates'].append([float(row_data[num_objects * 2 + i * 4]), float(row_data[num_objects * 2 + i * 4 + 1]), float(row_data[num_objects * 2 + i * 4 + 2]), float(row_data[num_objects * 2 + i * 4 + 3])])
                self.data['labels'].append(int(row_data[i * 2 + 1]))

        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.data['index'])


    def __getitem__(self, idx):
        index = self.data['index'][idx]

        image_folder = os.path.join(self.img_dir, str(index + 1))
        image_paths = os.listdir(image_folder)
        image_paths = natsorted(image_paths)
        # TODO: add image interval option
        image_paths = [os.path.join(image_folder, i) for i in image_paths]
        images = [self.transform(io.imread(i)) for i in image_paths]

        return images, self.data['coordinates'][idx], self.data['labels'][idx]