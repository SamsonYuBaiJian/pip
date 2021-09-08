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

    for _, video in enumerate(videos):
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
        color_data = sample_data[0]
        obj_color_dict = {}
        unused_obj = []
        for j in range(0, len(color_data), 2):
            obj_color_dict[color_data[j+1]] = color_data[j]
        sample_data = sample_data[1:]
        for obj in sample_data:
            obj_name = str(obj[0])
            initial_coordinates = [float(obj[3]), float(obj[5]), float(obj[7]), float(obj[9]), float(obj[11]), float(obj[13])]
            final_coordinates = [float(obj[-11]), float(obj[-9]), float(obj[-7]), float(obj[-5]), float(obj[-3]), float(obj[-1])]
            if task_type == 'contact':
                # skip red ball
                red_ball_initial_coordinates = [5, 5, 2, 0, 0, 0]
                if initial_coordinates == red_ball_initial_coordinates:
                    continue
            elif task_type == 'contain':
                # skip object holders
                object_holders = ['Pot_v1_001', 'Cylinder001', 'Glass', 'CardboardBox1', 'revolvedSurface1']
                if obj_name in object_holders:
                    del obj_color_dict[obj_name]
                    continue
                cube_initial_coordinates = [0, 0, -0.7720739841461182, 0, 0, 0]
                if initial_coordinates == cube_initial_coordinates:
                    continue
            # # ignore object name
            # for j in range(1, len(obj), 13):
            #     obj_coordinates.append([float(obj[j+2]), float(obj[j+4]), float(obj[j+6]), float(obj[j+8]), float(obj[j+10]), float(obj[j+12])])
            # get classification label for each task
            if task_type == 'contact':
                # if last x-coordinate or y-coordinate changes significantly
                if abs(initial_coordinates[0] - final_coordinates[0]) >= 0.2 or abs(initial_coordinates[1] - final_coordinates[1]) >= 0.2:
                    obj_label = [1]
                else:
                    obj_label = [0]
            elif task_type == 'contain':
                # if last z-coordinate is < 0
                if final_coordinates[2] < 0:
                    obj_label = [0]
                else:
                    obj_label = [1]
            elif task_type == 'stability':
                # if any of the rotation changes > 0.1
                if abs(initial_coordinates[3] - final_coordinates[3]) >= 0.1 or abs(initial_coordinates[4] - final_coordinates[4]) >= 0.1 or abs(initial_coordinates[5] - final_coordinates[5]) >= 0.1:
                    obj_label = [0]
                else:
                    obj_label = [1]
            if obj_name in obj_color_dict.keys():
                color = obj_color_dict[obj_name]
                output_data[sample_num].append([obj_name, color, obj_label])
                del obj_color_dict[obj_name]
            else:
                unused_obj.append([obj_name, obj_label])
            
        if len(unused_obj) > 0:
            if task_type == 'contain':
                if len(unused_obj) == 1 and len(obj_color_dict.keys()) == 1:
                    output_data[sample_num].append([unused_obj[0][0], obj_color_dict['Torus.020'], unused_obj[0][1]])
                else:
                    print("WARNING:", unused_obj, obj_color_dict, color_data)
            elif task_type == 'contact':
                if len(unused_obj) == 1 and len(obj_color_dict.keys()) == 2:
                    output_data[sample_num].append([unused_obj[0][0], obj_color_dict['Torus.24'], unused_obj[0][1]])
                else:
                    print("WARNING:", unused_obj, obj_color_dict, color_data)

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


    # data = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    # save_folder = "/home/samson/SPECIAL/dataset/contact/fixed_data"
    # for i in range(len(data)):
    #     data_name = data[i].split('/')[-1].split('.')[0]

    #     with open(data[i], 'r') as f:
    #         sample_data = ast.literal_eval(f.readline())
    #         f.close()

    #     num_obj = len(sample_data)

    #     objects = []
    #     all_data = []
    #     cnt = 0
    #     for j in range(len(sample_data)):
    #         all_data += sample_data[j]

    #     for j in all_data:
    #         if j == '1':
    #             cnt -= 1
    #             objects = objects[:-1]
    #             break
    #         else:
    #             cnt += 1
    #             objects.append(j)

        
    #     rest_data = all_data[cnt:]

    #     print(num_obj, len(objects), len(rest_data))

    #     # assert num_obj == len(objects) / 2

    #     assert(len(rest_data)) == (num_obj) * (150 * 13 + 1)

    #     final_data = []
    #     final_data.append(objects)
        
    #     for j in range(0,len(rest_data),150*13+1):
    #         # if rest_data[j] != 'Cube':
    #         final_data.append(rest_data[j:j+(150*13+1)])
        
    #     with open(save_folder + '/{}.txt'.format(data_name), 'w') as f:
    #         f.write(str(final_data))
    #         f.close()


    print("Processing data for {} task...".format(task_type))
    print("Getting data splits from {}...".format(data_path))
    get_dataset_splits(data_path, video_path, train_val_test_splits, task_type, train_label_path, val_label_path, test_label_path)
    print("Done.")
    # print("Getting frames from {} and saving to {}...".format(video_path, frame_path))
    # convert_avi_to_frame(video_path, frame_path)
    # print("Done.")