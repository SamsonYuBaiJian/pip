from torch.utils.data import Dataset
import os
from skimage import io
from torchvision import transforms
from natsort import natsorted
import numpy as np
import json
from transformers import BertTokenizer


class Data(Dataset):
    def __init__(self, frame_path, mask_path, label_path, frame_interval, first_n_frame_dynamics, task_type, model_type, max_seq_len, combined_scene_tasks=None):
        self.data = {'scene_id': [], 'color': [], 'labels': [], 'shape': []}
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            f.close()

        for scene_id in label_data.keys():
            scene_data = label_data[scene_id]
            num_objects = len(scene_data)
            for i in range(num_objects):
                self.data['scene_id'].append(scene_id)
                scene_data_i = scene_data[i]
                label = int(scene_data_i[2][0])
                self.data['color'].append(scene_data_i[1])
                self.data['labels'].append(label)
                self.data['shape'].append(scene_data_i[0].split('.')[0])

        self.frame_path = frame_path
        self.mask_path = mask_path
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        self.frame_interval = frame_interval
        self.first_n_frame_dynamics = first_n_frame_dynamics
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.task_type = task_type
        self.combined_scene_tasks = combined_scene_tasks
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def __len__(self):
        return len(self.data['scene_id'])


    def __getitem__(self, idx):
        scene_id = self.data['scene_id'][idx]
        color = self.data['color'][idx]
        shape = self.data['shape'][idx]

        image_folder = os.path.join(self.frame_path, scene_id)
        image_paths = os.listdir(image_folder)
        image_paths = natsorted(image_paths)
        final_image_paths = []
        # frame interval
        for i in range(0, len(image_paths), self.frame_interval):
            final_image_paths.append(os.path.join(image_folder, image_paths[i]))
        if self.model_type == 'pip_3':
            final_image_paths = final_image_paths[:self.first_n_frame_dynamics+1]
        else:
            final_image_paths = final_image_paths[:self.max_seq_len]
        assert len(final_image_paths) > 0
        images = [self.transform(io.imread(i)) for i in final_image_paths]

        mask_folder = os.path.join(self.mask_path, scene_id)
        mask_paths = os.listdir(mask_folder)
        mask_paths = natsorted(mask_paths)
        final_mask_paths = []
        for i in range(0, len(mask_paths), self.frame_interval):
            final_mask_paths.append(os.path.join(mask_folder, mask_paths[i]))
        final_mask_paths = final_mask_paths[:self.first_n_frame_dynamics+1]
        assert len(final_mask_paths) > 0
        masks = [self.transform(io.imread(i + '/{}.jpg'.format(color))) for i in final_mask_paths]

        if self.task_type == 'combined':
            for k,v in self.combined_scene_tasks.items():
                if int(scene_id) >= v[0] and int(scene_id) <= v[1]:
                    specific_task = k
                    break
            if specific_task == 'contact':
                query = "Does the {} {} get contacted by the red ball?".format(color, shape)
            elif specific_task == 'contain':
                query = "Is the {} {} contained within the containment holders?".format(color, shape)
            elif specific_task == 'stability':
                query = "Is the {} {} stable after it falls?".format(color, shape)
        else:
            if self.task_type == 'contact':
                query = "Does the {} {} get contacted by the red ball?".format(color, shape)
            elif self.task_type == 'contain':
                query = "Is the {} {} contained within the containment holder?".format(color, shape)
            elif self.task_type == 'stability':
                query = "Is the {} {} stable after it falls?".format(color, shape)
        encoded_query = self.tokenizer(query, padding='max_length', max_length=20, return_tensors='pt')

        return images, masks, self.data['labels'][idx], encoded_query