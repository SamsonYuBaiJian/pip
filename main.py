import torch.nn as nn
import torch
from dataloader import Data
from model import Generator
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import random
from piqa import SSIM, PSNR, MS_SSIM
import yaml
import argparse
import datetime
import os
import pandas as pd
import math


def get_classification_accuracy(pred_labels, labels):
    """
    Get accuracy for classification.
    """
    size = pred_labels.shape[0]
    mask = pred_labels >= 0.5
    num_correct = torch.sum(mask == labels).item()
    acc = num_correct / size

    return acc, num_correct


def main(cfg, task_type, frame_path, train_label_path, val_label_path, test_label_path, save_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, discriminator_window, learning_rate, save_frames_every):
    # get experiment ID
    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    experiment_save_path = os.path.join(save_path, experiment_id)
    os.makedirs(experiment_save_path, exist_ok=True)
    
    train_dataset = Data(frame_path, train_label_path, frame_interval)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataset = Data(frame_path, val_label_path, frame_interval)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # NOTE
    save_img_dir = os.path.join(experiment_save_path, 'generations')

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    generator = Generator(first_n_frame_dynamics).to(device)
    # turn off gradients for other tasks
    tasks = ['contact', 'contain', 'stability']
    for i in tasks:
        if i != task_type:
            for n, p in generator.named_parameters():
                if i in n:
                    p.requires_grad = False

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    bce_logits_loss = nn.BCEWithLogitsLoss().to(device)
    # image_loss = nn.MSELoss().to(device)
    # image_loss = SSIM().to(device)
    coordinate_mse_loss = nn.MSELoss().to(device)
    image_loss = MS_SSIM().to(device)

    stats = {'train': {'classification_loss': [], 'classification_acc': [], 'image_loss': [], 'coordinate_loss': []}, 'val': {'classification_loss': [], 'classification_acc': [], 'image_loss': [], 'coordinate_loss': []}}
    
    min_val_classification_loss = math.inf
    min_val_image_loss = math.inf
    min_val_coordinate_loss = math.inf
    min_val_classification_epoch = None
    min_val_image_epoch = None
    min_val_coordinate_epoch = None

    for i in range(num_epoch):
        # training
        print('Training for epoch {}/{}...'.format(i+1, num_epoch))
        temp_train_classification_loss = []
        temp_train_image_loss = []
        temp_train_coor_loss = []
        # temp_train_gen_adversarial_loss = []
        # temp_train_dis_adversarial_loss = []
        total_num_correct = 0
        total_cnt = 0
        for j, batch in tqdm(enumerate(train_dataloader)):
            generator.train()
            frames, coordinates, labels = batch
            # pass through model
            retrieved_batch_size = len(frames[0])
            total_cnt += retrieved_batch_size
            teacher_forcing_batch = random.choices(population=[True, False], weights=[teacher_forcing_prob, 1-teacher_forcing_prob], k=retrieved_batch_size)
            pred_labels, pred_images_seq, pred_coordinates_seq = generator(task_type, coordinates, frames, teacher_forcing_batch, first_n_frame_dynamics, device)
            labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
            train_acc, num_correct = get_classification_accuracy(pred_labels, labels)
            total_num_correct += num_correct
            bce_loss = bce_logits_loss(pred_labels, labels)
            gen_loss = bce_loss
            temp_train_classification_loss.append(bce_loss.data.item() * retrieved_batch_size)
            temp_train_image_loss.append(0)
            temp_train_coor_loss.append(0)

            # save generated images for testing
            if save_frames_every is not None and i % save_frames_every == 0:
                epoch_train_save_img_dir = os.path.join(os.path.join(save_img_dir, str(i)), 'train')
                pred_save_img_dir = os.path.join(epoch_train_save_img_dir, 'pred')
                real_save_img_dir = os.path.join(epoch_train_save_img_dir, 'real')
                os.makedirs(pred_save_img_dir, exist_ok=True)
                os.makedirs(real_save_img_dir, exist_ok=True)
                for k in range(first_n_frame_dynamics+1):
                    save_image(frames[k][0], os.path.join(pred_save_img_dir, '{}.png'.format(k)))
                    save_image(frames[k][0], os.path.join(real_save_img_dir, '{}.png'.format(k)))

            for k, pred_images in enumerate(pred_images_seq[:-1]):
                # save generated images for testing
                if save_frames_every is not None and i % save_frames_every == 0:
                    save_image(pred_images[0], os.path.join(pred_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
                    save_image(frames[k+first_n_frame_dynamics+1][0], os.path.join(real_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
                
                frames_k = frames[k+first_n_frame_dynamics+1].to(device)
                pred_images = torch.clamp(pred_images, 0, 1)
                img_loss = image_loss(pred_images, frames_k)
                gen_loss += - img_loss
                temp_train_image_loss[-1] += img_loss.data.item() * retrieved_batch_size
                seq_len = len(pred_images_seq[:-1])

                coordinates_k = torch.stack(coordinates[k+first_n_frame_dynamics+1]).T.to(device)
                coordinate_loss = coordinate_mse_loss(pred_coordinates_seq[k], coordinates_k.float())
                gen_loss += coordinate_loss
                temp_train_coor_loss[-1] += coordinate_loss.data.item() * retrieved_batch_size

            temp_train_image_loss[-1] /= seq_len
            temp_train_coor_loss[-1] /= seq_len
            
            if teacher_forcing_batch[0]:
                print("Saved new train frame sequences WITH teacher forcing.")
            else:
                print("Saved new train frame sequences WITHOUT teacher forcing.")

            generator.zero_grad()
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            print("Epoch {}/{} batch {}/{} training done with classification loss={}, classification accuracy={}, image loss={}, coordinate loss={}.".format(i+1, num_epoch, j+1, len(train_dataloader), temp_train_classification_loss[-1] / retrieved_batch_size, train_acc, temp_train_image_loss[-1] / retrieved_batch_size, temp_train_coor_loss[-1] / retrieved_batch_size)) # sum(temp_train_gen_adversarial_loss) / len(temp_train_gen_adversarial_loss), sum(temp_train_dis_adversarial_loss) / len(temp_train_dis_adversarial_loss)))

        print("Epoch {}/{} OVERALL train classification loss={}, classification accuracy={}, image loss={}, coordinate loss={}.".format(i+1, num_epoch, sum(temp_train_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_train_image_loss) / total_cnt, sum(temp_train_coor_loss) / total_cnt))
        stats['train']['classification_loss'].append(sum(temp_train_classification_loss) / total_cnt)
        stats['train']['classification_acc'].append(total_num_correct / total_cnt)
        stats['train']['image_loss'].append(sum(temp_train_image_loss) / total_cnt)
        stats['train']['coordinate_loss'].append(sum(temp_train_classification_loss) / total_cnt)


        # validation
        print('Validation for epoch {}/{}...'.format(i+1, num_epoch))
        temp_val_classification_loss = []
        temp_val_image_loss = []
        temp_val_coor_loss = []
        total_num_correct = 0
        total_cnt = 0
        generator.eval()
        with torch.no_grad():
            for j, batch in tqdm(enumerate(val_dataloader)):
                frames, coordinates, labels = batch
                retrieved_batch_size = len(frames[0])
                total_cnt += retrieved_batch_size
                # no teacher forcing for validation
                teacher_forcing_batch = random.choices(population=[True, False], weights=[0, 1], k=retrieved_batch_size)
                pred_labels, pred_images_seq, pred_coordinates_seq = generator(task_type, coordinates, frames, teacher_forcing_batch, first_n_frame_dynamics, device)
                labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
                val_acc, num_correct = get_classification_accuracy(pred_labels, labels)
                total_num_correct += num_correct
                bce_loss = bce_logits_loss(pred_labels, labels)
                # gen_loss = bce_loss
                temp_val_classification_loss.append(bce_loss.data.item() * retrieved_batch_size)
                temp_val_image_loss.append(0)
                temp_val_coor_loss.append(0)

                # save generated images for testing
                if save_frames_every is not None and i % save_frames_every == 0:
                    epoch_val_save_img_dir = os.path.join(os.path.join(save_img_dir, str(i)), 'val')
                    pred_save_img_dir = os.path.join(epoch_val_save_img_dir, 'pred')
                    real_save_img_dir = os.path.join(epoch_val_save_img_dir, 'real')
                    os.makedirs(pred_save_img_dir, exist_ok=True)
                    os.makedirs(real_save_img_dir, exist_ok=True)
                    for k in range(first_n_frame_dynamics+1):
                        save_image(frames[k][0], os.path.join(pred_save_img_dir, '{}.png'.format(k)))
                        save_image(frames[k][0], os.path.join(real_save_img_dir, '{}.png'.format(k)))

                for k, pred_images in enumerate(pred_images_seq[:-1]):
                    # save generated images for testing
                    if save_frames_every is not None and i % save_frames_every == 0:
                        save_image(pred_images[0], os.path.join(pred_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
                        save_image(frames[k+first_n_frame_dynamics+1][0], os.path.join(real_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
                    
                    frames_k = frames[k+first_n_frame_dynamics+1].to(device)
                    pred_images = torch.clamp(pred_images, 0, 1)
                    img_loss = image_loss(pred_images, frames_k)
                    temp_val_image_loss[-1] += img_loss.data.item() * retrieved_batch_size
                    # gen_loss += - img_loss
                    seq_len = len(pred_images_seq[:-1])

                    coordinates_k = torch.stack(coordinates[k+first_n_frame_dynamics+1]).T.to(device)
                    coordinate_loss = coordinate_mse_loss(pred_coordinates_seq[k], coordinates_k.float())
                    temp_val_coor_loss[-1] += coordinate_loss.data.item() * retrieved_batch_size
                temp_val_image_loss[-1] /= seq_len
                temp_val_coor_loss[-1] /= seq_len

                if teacher_forcing_batch[0]:
                    print("Saved new validation sequences WITH teacher forcing.")
                else:
                    print("Saved new validation sequences WITHOUT teacher forcing.")
                
                print("Epoch {}/{} batch {}/{} validation done with classification loss={}, classification accuracy={}, image loss={}, coordinate loss={}.".format(i+1, num_epoch, j+1, len(val_dataloader), temp_val_classification_loss[-1] / retrieved_batch_size, val_acc, temp_val_image_loss[-1] / retrieved_batch_size, temp_val_coor_loss[-1] / retrieved_batch_size))

        print("Epoch {}/{} OVERALL validation classification loss={}, classification accuracy={}, image loss={}, coordinate loss={}.".format(i+1, num_epoch, sum(temp_val_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_val_image_loss) / total_cnt, sum(temp_val_coor_loss) / total_cnt))
        stats['val']['classification_loss'].append(sum(temp_val_classification_loss) / total_cnt)
        stats['val']['classification_acc'].append(total_num_correct / total_cnt)
        stats['val']['image_loss'].append(sum(temp_val_image_loss) / total_cnt)
        stats['val']['coordinate_loss'].append(sum(temp_val_coor_loss) / total_cnt)

        if stats['val']['classification_loss'][-1] < min_val_classification_loss:
            min_val_classification_loss = stats['val']['classification_loss'][-1]
            min_val_classification_epoch = i
            torch.save(generator, os.path.join(experiment_save_path, 'classification_model'))
        if stats['val']['image_loss'][-1] < min_val_image_loss:
            min_val_image_loss = stats['val']['image_loss'][-1]
            min_val_image_epoch = i
            torch.save(generator, os.path.join(experiment_save_path, 'image_model'))
        if stats['val']['coordinate_loss'][-1] < min_val_coordinate_loss:
            min_val_coordinate_loss = stats['val']['coordinate_loss'][-1]
            min_val_coordinate_epoch = i
            torch.save(generator, os.path.join(experiment_save_path, 'coordinate_model'))

        with open(os.path.join(experiment_save_path, 'log.txt'), 'w') as f:
            f.write('{}\n'.format(cfg))
            f.write('{}\n'.format(stats))
            f.write('Min classification_loss: epoch {}, {}\n'.format(min_val_classification_epoch, min_val_classification_loss))
            f.write('Min image_loss: epoch {}, {}\n'.format(min_val_image_epoch, min_val_image_loss))
            f.write('Min coordinate_loss: epoch {}, {}'.format(min_val_coordinate_epoch, min_val_coordinate_loss))
            f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    args = parser.parse_args()
    with open(args.config_file, "r") as setting:
        cfg = yaml.safe_load(setting)

    # load config
    task_type = cfg['task_type']
    frame_path = cfg['frame_path']
    train_label_path = cfg['train_label_path']
    val_label_path = cfg['val_label_path']
    test_label_path = cfg['test_label_path']
    save_path = cfg['save_path']
    save_frames_every = cfg['save_frames_every']
    num_epoch = cfg['num_epoch']
    batch_size = cfg['batch_size']
    teacher_forcing_prob = cfg['teacher_forcing_prob']
    first_n_frame_dynamics = cfg['first_n_frame_dynamics']
    frame_interval = cfg['frame_interval']
    learning_rate = cfg['learning_rate']

    # check configs
    if task_type != 'contact' and task_type != 'contain' and task_type != 'stability':
        assert False, "Is your task_type contact, contain or stability?"
    assert num_epoch > 0 and type(num_epoch) == int
    assert batch_size > 0 and type(batch_size) == int
    assert teacher_forcing_prob >= 0 and teacher_forcing_prob <= 1
    assert first_n_frame_dynamics >= 0 and type(first_n_frame_dynamics) == int
    assert frame_interval > 0 and type(frame_interval) == int
    assert learning_rate > 0

    main(cfg, task_type, frame_path, train_label_path, val_label_path, test_label_path, save_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, None, learning_rate, save_frames_every)