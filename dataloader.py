import pandas as pd
from torch.utils.data import Dataset
import os
from skimage import io
from torchvision import transforms
from natsort import natsorted
import torch
import numpy as np


class Data(Dataset):
    def __init__(self, img_dir, csv_file, frame_interval, task_type):
        self.data = {'index': [], 'coordinates': [], 'labels': []}
        df = pd.read_csv(csv_file)
        for row in df.itertuples():
            # gets rid of NaN values
            row_data = [i for i in row if i == i]
            index = int(row_data[0])
            row_data = row_data[1:]
            if task_type == 'contact':
                col_per_obj = 6
            elif task_type == 'contain':
                col_per_obj = 4
            elif task_type == 'stability':
                col_per_obj = 5
            num_objects = int(len(row_data) / col_per_obj)
            for i in range(num_objects):
                self.data['index'].append(index)
                self.data['coordinates'].append([float(row_data[num_objects * 2 + i * (col_per_obj - 2) + j]) for j in range(col_per_obj - 2)])
                # self.data['coordinates'].append([float(row_data[num_objects * 2 + i * 4]), float(row_data[num_objects * 2 + i * 4 + 1]), float(row_data[num_objects * 2 + i * 4 + 2]), float(row_data[num_objects * 2 + i * 4 + 3])])
                self.data['labels'].append(int(row_data[i * 2 + 1]))

        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(),
            transforms.ToTensor(),
        ])
        self.frame_interval = frame_interval


    def __len__(self):
        return len(self.data['index'])


    def __getitem__(self, idx):
        index = self.data['index'][idx]

        image_folder = os.path.join(self.img_dir, str(index + 1))
        image_paths = os.listdir(image_folder)
        image_paths = natsorted(image_paths)
        # frame interval
        image_paths = [os.path.join(image_folder, image_paths[i]) for i in range(len(image_paths)) if i % self.frame_interval == 0]
        assert len(image_paths) > 0
        images = [self.transform(io.imread(i)) for i in image_paths]

        return images, self.data['coordinates'][idx], self.data['labels'][idx]