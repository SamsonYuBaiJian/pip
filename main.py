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


def get_classification_accuracy(pred_labels, labels):
    """
    Get accuracy for classification.
    """
    size = pred_labels.shape[0]
    mask = pred_labels >= 0.5
    num_correct = torch.sum(mask == labels).item()
    acc = num_correct / size

    return acc, num_correct


def main(cfg, task_type, frame_path, train_label_path, val_label_path, test_label_path, save_stats_path, save_generated_images_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, discriminator_window, learning_rate):
    # get experiment ID
    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path, exist_ok=True)
    
    train_dataset = Data(frame_path, train_label_path, frame_interval)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataset = Data(frame_path, val_label_path, frame_interval)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # NOTE
    pred_save_img_dir = os.path.join(save_generated_images_path, 'pred')
    real_save_img_dir = os.path.join(save_generated_images_path, 'real')
    os.makedirs(pred_save_img_dir, exist_ok=True)
    os.makedirs(real_save_img_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(first_n_frame_dynamics).to(device)
    # turn off gradients for other tasks
    tasks = ['contact', 'contain', 'stability']
    for i in tasks:
        if i != task_type:
            for n, p in generator.named_parameters():
                if i in n:
                    p.requires_grad = False
    # discriminator = Discriminator(discriminator_window).to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    # dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    bce_logits_loss = nn.BCEWithLogitsLoss().to(device)
    # image_loss = nn.MSELoss().to(device)
    # image_loss = SSIM().to(device)
    coordinate_mse_loss = nn.MSELoss().to(device)
    image_loss = MS_SSIM().to(device)

    stats = {'train': {'classification_loss': [], 'classification_acc': [], 'image_loss': [], 'coordinate_loss': []}, 'val': {'classification_loss': [], 'classification_acc': [], 'image_loss': [], 'coordinate_loss': []}}
    
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
            # discriminator.eval()
            frames, coordinates, labels = batch
            # pass through generator
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

            # NOTE: save generated images for testing
            for k in range(first_n_frame_dynamics+1):
                save_image(frames[k][0], os.path.join(pred_save_img_dir, '{}.png'.format(k)))
                save_image(frames[k][0], os.path.join(real_save_img_dir, '{}.png'.format(k)))

            for k, pred_images in enumerate(pred_images_seq[:-1]):
                # NOTE: save generated images for testing
                save_image(pred_images[0], os.path.join(pred_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
                save_image(frames[k+first_n_frame_dynamics+1][0], os.path.join(real_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
                
                frames_k = frames[k+first_n_frame_dynamics+1].to(device)
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
            # # pass through discriminator
            # for k in range(0, len(pred_images_seq[:-1]), discriminator_window):
            #     if k+discriminator_window-1 < len(pred_images_seq[:-1]):
            #         dis_pred_images_seq = pred_images_seq[:-1][k:k+discriminator_window]
            #         dis_frames_seq = frames[k+first_n_frame_dynamics+1:k+first_n_frame_dynamics+discriminator_window+1]
            #         dis_pred = discriminator(dis_pred_images_seq, dis_frames_seq, device)
            #         dis_labels = torch.zeros_like(dis_pred)
            #         dis_labels[retrieved_batch_size:,:,:,:] = 1.
            #         adversarial_loss = bce_logits_loss(dis_pred, dis_labels)
            #         temp_train_gen_adversarial_loss.append(adversarial_loss.data.item())
            #         gen_loss += adversarial_loss
            generator.zero_grad()
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # # train discriminator and freeze generator
            # generator.eval()
            # discriminator.train()
            # # NOTE: all non-teacher forcing for discriminator?
            # # teacher_forcing_batch = [False] * retrieved_batch_size
            # _, pred_images_seq = generator(task_type, coordinates, frames, teacher_forcing_batch, first_n_frame_dynamics, device)
            # dis_loss = 0
            # for k in range(0, len(pred_images_seq[:-1]), discriminator_window):
            #     if k+discriminator_window-1 < len(pred_images_seq[:-1]):
            #         dis_pred_images_seq = pred_images_seq[:-1][k:k+discriminator_window]
            #         dis_frames_seq = frames[k+first_n_frame_dynamics+1:k+first_n_frame_dynamics+discriminator_window+1]
            #         dis_pred = discriminator(dis_pred_images_seq, dis_frames_seq, device)
            #         dis_labels = torch.zeros_like(dis_pred)
            #         dis_labels[retrieved_batch_size:,:,:,:] = 1.
            #         adversarial_loss = bce_logits_loss(dis_pred, dis_labels)
            #         temp_train_dis_adversarial_loss.append(adversarial_loss.data.item())
            #         dis_loss += adversarial_loss
            # discriminator.zero_grad()
            # dis_optimizer.zero_grad()
            # dis_loss.backward()
            # dis_optimizer.step()

            print("Epoch {}/{} batch {}/{} training done with classification loss={}, classification accuracy={}, image loss={}, coordinate loss={}.".format(i+1, num_epoch, j+1, len(train_dataloader), temp_train_classification_loss[-1] / retrieved_batch_size, train_acc, temp_train_image_loss[-1] / retrieved_batch_size, temp_train_coor_loss[-1] / retrieved_batch_size)) # sum(temp_train_gen_adversarial_loss) / len(temp_train_gen_adversarial_loss), sum(temp_train_dis_adversarial_loss) / len(temp_train_dis_adversarial_loss)))

        print("Epoch {}/{} OVERALL train classification loss={}, classification accuracy={}, image loss={}, coordinate loss={}.".format(i+1, num_epoch, sum(temp_train_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_train_image_loss) / total_cnt, sum(temp_train_coor_loss) / total_cnt))
        stats['train']['classification_loss'].append(sum(temp_train_classification_loss) / total_cnt)
        stats['train']['classification_acc'].append(total_num_correct / total_cnt)
        stats['train']['image_loss'].append(sum(temp_train_image_loss) / total_cnt)
        stats['train']['coordinate_loss'].append(sum(temp_train_classification_loss) / total_cnt)
        # stats['train']['gen_adversarial_loss'].append(sum(temp_train_gen_adversarial_loss) / len(temp_train_gen_adversarial_loss))
        # stats['train']['dis_adversarial_loss'].append(sum(temp_train_dis_adversarial_loss) / len(temp_train_dis_adversarial_loss))


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

                # NOTE: save generated images for testing
                for k in range(first_n_frame_dynamics+1):
                    save_image(frames[k][0], os.path.join(pred_save_img_dir, '{}.png'.format(k)))
                    save_image(frames[k][0], os.path.join(real_save_img_dir, '{}.png'.format(k)))

                for k, pred_images in enumerate(pred_images_seq[:-1]):
                    # NOTE: save generated images for testing
                    save_image(pred_images[0], os.path.join(pred_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
                    save_image(frames[k+first_n_frame_dynamics+1][0], os.path.join(real_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
                    
                    frames_k = frames[k+first_n_frame_dynamics+1].to(device)
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

        with open(os.path.join(save_stats_path, '{}.txt'.format(experiment_id)), 'w') as f:
            f.write('{}\n'.format(cfg))
            f.write('{}'.format(stats))
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
    save_stats_path = cfg['save_stats_path']
    save_generated_images_path = cfg['save_generated_images_path']
    num_epoch = cfg['num_epoch']
    batch_size = cfg['batch_size']
    teacher_forcing_prob = cfg['teacher_forcing_prob']
    first_n_frame_dynamics = cfg['first_n_frame_dynamics']
    frame_interval = cfg['frame_interval']
    # discriminator_window = cfg['discriminator_window']
    learning_rate = cfg['learning_rate']

    # check configs
    if task_type != 'contact' and task_type != 'contain' and task_type != 'stability':
        assert False, "Is your task_type contact, contain or stability?"
    assert num_epoch > 0 and type(num_epoch) == int
    assert batch_size > 0 and type(batch_size) == int
    assert teacher_forcing_prob >= 0 and teacher_forcing_prob <= 1
    assert first_n_frame_dynamics >= 0 and type(first_n_frame_dynamics) == int
    assert frame_interval > 0 and type(frame_interval) == int
    # assert discriminator_window > 0 and type(discriminator_window) == int
    assert learning_rate > 0

    main(cfg, task_type, frame_path, train_label_path, val_label_path, test_label_path, save_stats_path, save_generated_images_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, None, learning_rate)