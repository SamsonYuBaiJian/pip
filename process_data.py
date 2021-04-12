import cv2
import os
import argparse
import yaml
import ast
import json
from natsort import natsorted


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


def save_data(data, start_idx, end_idx, label_path, task_type):
    """
    Save information for data splits into .json files.
    """
    output_data = {}
    for i in range(start_idx, end_idx):
        with open(data[i], 'r') as f:
            sample_data = ast.literal_eval(f.readline())
            f.close()
        sample_num = int(i+1)
        output_data[sample_num] = []
        # for each object for object tracking
        for obj in sample_data:
            obj_coordinates = []
            # TODO: ignore red ball for each task
            initial_coordinates = [float(obj[3]), float(obj[5]), float(obj[7]), float(obj[9]), float(obj[11]), float(obj[13])]
            if task_type == 'contact':
                red_ball_initial_coordinates = [5, 5, 2, 0, 0, 0]
            if initial_coordinates == red_ball_initial_coordinates:
                # skip if red ball
                continue
            # ignore object name
            for j in range(1, len(obj), 13):
                obj_coordinates.append([float(obj[j+2]), float(obj[j+4]), float(obj[j+6]), float(obj[j+8]), float(obj[j+10]), float(obj[j+12])])
            # TODO: get classification label for each task
            if task_type == 'contact':
                if abs(obj_coordinates[0][0] - obj_coordinates[-1][0]) >= 0.05 or abs(obj_coordinates[0][1] - obj_coordinates[-1][1]) >= 0.05:
                    obj_coordinates.append([1])
                else:
                    obj_coordinates.append([0])
            output_data[sample_num].append(obj_coordinates)
    with open(label_path, 'w') as fp:
        json.dump(output_data, fp)
        fp.close()


def get_dataset_splits(data_path, video_path, train_val_test_splits, task_type, train_label_path, val_label_path, test_label_path):
    """
    Get dataset splits from initial data.
    """
    data = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    data = natsorted(data)
    data_num = len(data)
    video_num = len([os.path.join(video_path, i) for i in os.listdir(video_path)])

    assert data_num == video_num, "Number of data samples not the same as number of video files. There are {} data samples, and {} video files.".format(data_num, video_num)
    train_part, val_part, _ = train_val_test_splits
    train_num = int(data_num * train_part)
    val_num = int(data_num * val_part)

    # save splits
    save_data(data, 0, train_num, train_label_path, task_type)
    save_data(data, train_num, int(train_num + val_num), val_label_path, task_type)
    save_data(data, int(train_num + val_num), int(data_num), test_label_path, task_type)


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
    data_path = cfg['data_path']
    train_val_test_splits = cfg['train_val_test_splits']
    task_type = cfg['task_type']
    
    if task_type != 'contact' and task_type != 'contain' and task_type != 'stability':
        assert False, "Is your task_type contact, contain or stability?"
    assert len(train_val_test_splits) == 3 and sum(train_val_test_splits) == 1

    print("Processing data for {} task...".format(task_type))
    print("Getting data splits from {}...".format(data_path))
    get_dataset_splits(data_path, video_path, train_val_test_splits, task_type, train_label_path, val_label_path, test_label_path)
    print("Done.")
    print("Getting frames from {} and saving to {}...".format(video_path, frame_path))
    convert_avi_to_frame(video_path, frame_path)
    print("Done.")