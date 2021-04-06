import torch.nn as nn
import torch
from dataloader import Data
from model import Discriminator, Generator
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


def main(cfg, task_type, frame_path, label_path, save_stats_path, save_generated_images_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, discriminator_window, learning_rate, train_val_split):
    # get experiment ID
    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path, exist_ok=True)
    
    # process train-val split
    df = pd.read_csv(label_path)
    df_len = len(df.index)
    train_size = int(train_val_split * df_len)
    train_indices = [i for i in range(train_size)]
    train_dataset = Data(frame_path, label_path, train_indices, frame_interval, task_type)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_size = df_len - train_size
    first_val_index = train_indices[-1] + 1
    val_indices = [first_val_index+i for i in range(val_size)]
    val_dataset = Data(frame_path, label_path, val_indices, frame_interval, task_type)
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
    discriminator = Discriminator(discriminator_window).to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    bce_logits_loss = nn.BCEWithLogitsLoss().to(device)
    # image_loss = nn.MSELoss().to(device)
    # image_loss = SSIM().to(device)
    image_loss = MS_SSIM().to(device)

    stats = {'train': {'bce_loss': [], 'image_loss': [], 'gen_adversarial_loss': [], 'dis_adversarial_loss': []}, 'val': {'bce_loss': [], 'image_loss': []}}
    
    for i in range(num_epoch):
        # training
        print('Training for epoch {}/{}...'.format(i+1, num_epoch))
        temp_train_bce_loss = []
        temp_train_image_loss = []
        temp_train_gen_adversarial_loss = []
        temp_train_dis_adversarial_loss = []
        for j, batch in tqdm(enumerate(train_dataloader)):
            # train generator and freeze discriminator
            generator.train()
            discriminator.eval()
            frames, coordinates, labels = batch
            # pass through generator
            retrieved_batch_size = len(frames[0])
            teacher_forcing_batch = random.choices(population=[True, False], weights=[teacher_forcing_prob, 1-teacher_forcing_prob], k=retrieved_batch_size)
            pred_labels, pred_images_seq = generator(task_type, coordinates, frames, teacher_forcing_batch, first_n_frame_dynamics, device)
            labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
            bce_loss = bce_logits_loss(pred_labels, labels)
            temp_train_bce_loss.append(bce_loss.data.item())
            gen_loss = bce_loss

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
                temp_train_image_loss.append(img_loss.data.item())
                gen_loss += - img_loss
            if teacher_forcing_batch[0]:
                print("Saved new train frame sequences WITH teacher forcing.")
            else:
                print("Saved new train frame sequences WITHOUT teacher forcing.")
            # pass through discriminator
            for k in range(0, len(pred_images_seq[:-1]), discriminator_window):
                if k+discriminator_window-1 < len(pred_images_seq[:-1]):
                    dis_pred_images_seq = pred_images_seq[:-1][k:k+discriminator_window]
                    dis_frames_seq = frames[k+first_n_frame_dynamics+1:k+first_n_frame_dynamics+discriminator_window+1]
                    dis_pred = discriminator(dis_pred_images_seq, dis_frames_seq, device)
                    dis_labels = torch.zeros_like(dis_pred)
                    dis_labels[retrieved_batch_size:,:,:,:] = 1.
                    adversarial_loss = bce_logits_loss(dis_pred, dis_labels)
                    temp_train_gen_adversarial_loss.append(adversarial_loss.data.item())
                    gen_loss += adversarial_loss
            generator.zero_grad()
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # train discriminator and freeze generator
            generator.eval()
            discriminator.train()
            # NOTE: all non-teacher forcing for discriminator?
            teacher_forcing_batch = [False] * retrieved_batch_size
            _, pred_images_seq = generator(task_type, coordinates, frames, teacher_forcing_batch, first_n_frame_dynamics, device)
            dis_loss = 0
            for k in range(0, len(pred_images_seq[:-1]), discriminator_window):
                if k+discriminator_window-1 < len(pred_images_seq[:-1]):
                    dis_pred_images_seq = pred_images_seq[:-1][k:k+discriminator_window]
                    dis_frames_seq = frames[k+first_n_frame_dynamics+1:k+first_n_frame_dynamics+discriminator_window+1]
                    dis_pred = discriminator(dis_pred_images_seq, dis_frames_seq, device)
                    dis_labels = torch.zeros_like(dis_pred)
                    dis_labels[retrieved_batch_size:,:,:,:] = 1.
                    adversarial_loss = bce_logits_loss(dis_pred, dis_labels)
                    temp_train_dis_adversarial_loss.append(adversarial_loss.data.item())
                    dis_loss += adversarial_loss
            discriminator.zero_grad()
            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()

            print("Epoch {}/{} batch {}/{} training done with average BCE loss={}, image loss={}, generator adversarial loss={} and discriminator adversarial loss={}.".format(i+1, num_epoch, j+1, len(train_dataloader), sum(temp_train_bce_loss) / len(temp_train_bce_loss), sum(temp_train_image_loss) / len(temp_train_image_loss), sum(temp_train_gen_adversarial_loss) / len(temp_train_gen_adversarial_loss), sum(temp_train_dis_adversarial_loss) / len(temp_train_dis_adversarial_loss)))

        print("Epoch {}/{} OVERALL train BCE loss={} and image loss={}".format(i+1, num_epoch, sum(temp_train_bce_loss) / len(temp_train_bce_loss), sum(temp_train_image_loss) / len(temp_train_image_loss)))
        stats['train']['bce_loss'].append(sum(temp_train_bce_loss) / len(temp_train_bce_loss))
        stats['train']['image_loss'].append(sum(temp_train_image_loss) / len(temp_train_image_loss))
        stats['train']['gen_adversarial_loss'].append(sum(temp_train_gen_adversarial_loss) / len(temp_train_gen_adversarial_loss))
        stats['train']['dis_adversarial_loss'].append(sum(temp_train_dis_adversarial_loss) / len(temp_train_dis_adversarial_loss))


        # validation
        print('Validation for epoch {}/{}...'.format(i+1, num_epoch))
        temp_val_bce_loss = []
        temp_val_image_loss = []
        generator.eval()
        with torch.no_grad():
            for j, batch in tqdm(enumerate(val_dataloader)):
                frames, coordinates, labels = batch
                retrieved_batch_size = len(frames[0])
                # no teacher forcing for validation
                teacher_forcing_batch = random.choices(population=[True, False], weights=[0, 1], k=retrieved_batch_size)
                pred_labels, pred_images_seq = generator(task_type, coordinates, frames, teacher_forcing_batch, first_n_frame_dynamics, device)
                labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
                bce_loss = bce_logits_loss(pred_labels, labels)
                temp_val_bce_loss.append(bce_loss.data.item())
                gen_loss = bce_loss

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
                    temp_val_image_loss.append(img_loss.data.item())
                    gen_loss += - img_loss
                if teacher_forcing_batch[0]:
                    print("Saved new validation sequences WITH teacher forcing.")
                else:
                    print("Saved new validation sequences WITHOUT teacher forcing.")
                
                print("Epoch {}/{} batch {}/{} validation done with average BCE loss={} and image loss={}.".format(i+1, num_epoch, j+1, len(val_dataloader), sum(temp_val_bce_loss) / len(temp_val_bce_loss), sum(temp_val_image_loss) / len(temp_val_image_loss)))

        print("Epoch {}/{} OVERALL validation BCE loss={} and image loss={}".format(i+1, num_epoch, sum(temp_val_bce_loss) / len(temp_val_bce_loss), sum(temp_val_image_loss) / len(temp_val_image_loss)))
        stats['val']['bce_loss'].append(sum(temp_val_bce_loss) / len(temp_val_bce_loss))
        stats['val']['image_loss'].append(sum(temp_val_image_loss) / len(temp_val_image_loss))

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
    label_path = cfg['label_path']
    save_stats_path = cfg['save_stats_path']
    save_generated_images_path = cfg['save_generated_images_path']
    num_epoch = cfg['num_epoch']
    batch_size = cfg['batch_size']
    teacher_forcing_prob = cfg['teacher_forcing_prob']
    first_n_frame_dynamics = cfg['first_n_frame_dynamics']
    frame_interval = cfg['frame_interval']
    discriminator_window = cfg['discriminator_window']
    learning_rate = cfg['learning_rate']
    train_val_split = cfg['train_val_split']

    # check configs
    if task_type != 'contact' and task_type != 'contain' and task_type != 'stability':
        assert False, "Is your task_type contact, contain or stability?"
    assert num_epoch > 0 and type(num_epoch) == int
    assert batch_size > 0 and type(batch_size) == int
    assert teacher_forcing_prob >= 0 and teacher_forcing_prob <= 1
    assert first_n_frame_dynamics >= 0 and type(first_n_frame_dynamics) == int
    assert frame_interval > 0 and type(frame_interval) == int
    assert discriminator_window > 0 and type(discriminator_window) == int
    assert learning_rate > 0
    assert train_val_split >= 0 and train_val_split <=1

    main(cfg, task_type, frame_path, label_path, save_stats_path, save_generated_images_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, discriminator_window, learning_rate, train_val_split)