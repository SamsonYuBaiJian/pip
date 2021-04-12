from torch.utils.data import Dataset
import os
from skimage import io
from torchvision import transforms
from natsort import natsorted
import numpy as np
import json


class Data(Dataset):
    def __init__(self, frame_path, label_path, frame_interval):
        self.data = {'scene_id': [], 'coordinates': [], 'labels': []}
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            f.close()

        # process frame interval skips here to reduce cost for retrieval
        for scene_id in label_data.keys():
            total_seq_len = len(label_data[scene_id][0]) - 1
            break
        self.frame_indices = [i for i in range(total_seq_len) if i % frame_interval == 0]

        for scene_id in label_data.keys():
            scene_data = label_data[scene_id]
            num_objects = len(scene_data)
            for i in range(num_objects):
                self.data['scene_id'].append(scene_id)
                scene_data_i = scene_data[i]
                coordinates = scene_data_i[:-1]
                label = int(scene_data_i[-1][0])
                self.data['coordinates'].append([coordinates[j] for j in self.frame_indices])
                self.data['labels'].append(label)

        self.frame_path = frame_path
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(),
            transforms.ToTensor(),
        ])
        self.frame_interval = frame_interval


    def __len__(self):
        return len(self.data['scene_id'])


    def __getitem__(self, idx):
        scene_id = self.data['scene_id'][idx]

        image_folder = os.path.join(self.frame_path, scene_id)
        image_paths = os.listdir(image_folder)
        image_paths = natsorted(image_paths)
        # frame interval
        image_paths = [os.path.join(image_folder, image_paths[i]) for i in self.frame_indices]
        assert len(image_paths) > 0
        images = [self.transform(io.imread(i)) for i in image_paths]

        return images, self.data['coordinates'][idx], self.data['labels'][idx]