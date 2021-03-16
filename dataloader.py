import pandas as pd
from torch.utils.data import Dataset
import os
from skimage import io
from torchvision import transforms


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
                self.data['coordinates'].append([row_data[num_objects * 2 + i * 4], row_data[num_objects * 2 + i * 4 + 1], row_data[num_objects * 2 + i * 4 + 2], row_data[num_objects * 2 + i * 4 + 3]])
                self.data['labels'].append(int(row_data[i * 2 + 1]))

        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            # transforms.CenterCrop(),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        indices = self.data['index'][idx]

        frames = []
        try:
            for index in indices:
                image_folder = os.path.join(self.img_dir, str(index + 1))
                image_paths = os.listdir(image_folder)
                image_paths = [os.path.join(image_folder, i) for i in image_paths]
                images = [self.transform(io.imread(i)) for i in image_paths]
                frames.append(images)
        except TypeError:
            image_folder = os.path.join(self.img_dir, str(indices + 1))
            image_paths = os.listdir(image_folder)
            image_paths = [os.path.join(image_folder, i) for i in image_paths]
            images = [self.transform(io.imread(i)) for i in image_paths]
            frames.append(images)

        sample = {'frames': frames, 'coordinates': self.data['coordinates'][idx], 'labels': self.data['labels'][idx]}

        return sample