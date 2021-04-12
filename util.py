import cv2
import os
import argparse
import yaml
import ast
import json


def convert_avi_to_frame(video_path, frame_path):
    """
    Get frames from .avi video files.
    """
    os.makedirs(frame_path, exist_ok=True)

    videos = [os.path.join(video_path, i) for i in os.listdir(video_path)]

    for video in videos:
        video_id = video.split('/')[-1].split('.')[0]
        video_folder = os.path.join(frame_path, video_id)
        os.makedirs(video_folder, exist_ok=True)
        video_cap = cv2.VideoCapture(video)
        success, image = video_cap.read()
        count = 0
        while success:
            # save frame as JPEG file
            cv2.imwrite(os.path.join(video_folder, "frame_{}.jpg".format(count)), image)    
            success, image = video_cap.read()
            count += 1


def get_dataset_splits(data_txt_file, video_path, train_val_test_splits, task_type, train_label_path, val_label_path, test_label_path):
    """
    Get dataset splits from original text file.
    """
    with open(data_txt_file, 'r') as f:
        data = ast.literal_eval(f.readline())
        f.close()
    
    data_num = len(data) / 2
    video_num = len([os.path.join(video_path, i) for i in os.listdir(video_path)])

    assert data_num == video_num, "Number of data samples not the same as number of video files. There are {} data samples, and {} video files.".format(data_num, video_num)

    train_part, val_part, _ = train_val_test_splits
    train_num = int(data_num * train_part)
    val_num = int(data_num * val_part)

    # get train split
    train_data = {}
    for i in range(0, train_num, 2):
        sample_num = int(data[i])
        train_data[sample_num] = []
        sample_data = ast.literal_eval(data[i+1])
        # for each object for object tracking
        for obj in sample_data:
            obj_coordinates = []
            # ignore object name
            for j in range(1, len(obj), 13):
                obj_coordinates.append([float(obj[j+2]), float(obj[j+4]), float(obj[j+6]), float(obj[j+8]), float(obj[j+10]), float(obj[j+12])])
            train_data[sample_num].append(obj_coordinates)
    with open(train_label_path, 'w') as fp:
        json.dump(train_data, fp)
        fp.close()

    # get val split
    val_data = {}
    for i in range(train_num, int(train_num + val_num), 2):
        sample_num = int(data[i])
        val_data[sample_num] = []
        sample_data = ast.literal_eval(data[i+1])
        # for each object for object tracking
        for obj in sample_data:
            obj_coordinates = []
            # ignore object name
            for j in range(1, len(obj), 13):
                obj_coordinates.append([float(obj[j+2]), float(obj[j+4]), float(obj[j+6]), float(obj[j+8]), float(obj[j+10]), float(obj[j+12])])
            val_data[sample_num].append(obj_coordinates)
    with open(val_label_path, 'w') as fp:
        json.dump(val_data, fp)
        fp.close()

    # get test split
    test_data = {}
    for i in range(int(train_num + val_num), int(data_num), 2):
        sample_num = int(data[i])
        test_data[sample_num] = []
        sample_data = ast.literal_eval(data[i+1])
        # for each object for object tracking
        for obj in sample_data:
            obj_coordinates = []
            # ignore object name
            for j in range(1, len(obj), 13):
                obj_coordinates.append([float(obj[j+2]), float(obj[j+4]), float(obj[j+6]), float(obj[j+8]), float(obj[j+10]), float(obj[j+12])])
            test_data[sample_num].append(obj_coordinates)
    with open(test_label_path, 'w') as fp:
        json.dump(test_data, fp)
        fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    args = parser.parse_args()
    with open(args.config_file, "r") as setting:
        cfg = yaml.safe_load(setting)

    frame_path = cfg['frame_path']
    train_label_path = cfg['train_label_path']
    val_label_path = cfg['val_label_path']
    test_label_path = cfg['test_label_path']
    video_path = cfg['video_path']
    data_txt_file = cfg['data_txt_file']
    train_val_test_splits = cfg['train_val_test_splits']
    task_type = cfg['task_type']
    
    if task_type != 'contact' and task_type != 'contain' and task_type != 'stability':
        assert False, "Is your task_type contact, contain or stability?"
    assert len(train_val_test_splits) == 3 and sum(train_val_test_splits) == 1

    print("Getting data splits from {}...".format(data_txt_file))
    get_dataset_splits(data_txt_file, video_path, train_val_test_splits, task_type, train_label_path, val_label_path, test_label_path)
    print("Done.")
    print("Getting frames from {}...".format(video_path))
    convert_avi_to_frame(video_path, frame_path)
    print("Done.")