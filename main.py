import torch.nn as nn
import torch
from dataloader import Data
from model import Model
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import random


# NOTE: save generated images for testing
import os
test_img_dir = '/mnt/c/Users/samso/Desktop/test/'
pred_test_img_dir = os.path.join(test_img_dir, 'pred')
real_test_img_dir = os.path.join(test_img_dir, 'real')
os.makedirs(pred_test_img_dir, exist_ok=True)
os.makedirs(real_test_img_dir, exist_ok=True)


def main(task, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval):
    train_dataset = Data('/mnt/c/Users/samso/Documents/SamsonYuBaiJian/CLEVEREST/dataset/contact/labels.csv', '/mnt/c/Users/samso/Documents/SamsonYuBaiJian/CLEVEREST/dataset/contact/frames', frame_interval)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    # turn off gradients for other tasks
    tasks = ['contact', 'contain', 'stability']
    for i in tasks:
        if i != task:
            for n, p in model.named_parameters():
                if i in n:
                    p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    bce_logits_loss = nn.BCEWithLogitsLoss().to(device)
    mse_loss = nn.MSELoss().to(device)

    model.train()
    print('Training...')
    train_bce_loss_per_epoch = []
    train_mse_loss_per_epoch = []
    for i in range(num_epoch):
        temp_train_bce_loss = []
        temp_train_mse_loss = []
        for j, batch in tqdm(enumerate(train_dataloader)):
            frames, coordinates, labels = batch
            retrieved_batch_size = len(frames[0])
            teacher_forcing_batch = random.choices(population=[True, False], weights=[teacher_forcing_prob, 1-teacher_forcing_prob], k=retrieved_batch_size)
            pred_labels, pred_images_seq = model(task, coordinates, frames, teacher_forcing_batch, first_n_frame_dynamics, device)
            labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
            bce_loss = bce_logits_loss(pred_labels, labels)
            temp_train_bce_loss.append(bce_loss.data.item())
            loss = bce_loss
            frames = frames[1:]
            for k, pred_images in enumerate(pred_images_seq[:-1]):

                # NOTE: save generated images for testing
                save_image(pred_images[0], os.path.join(pred_test_img_dir, '{}.png'.format(k+1)))
                save_image(frames[k][0], os.path.join(real_test_img_dir, '{}.png'.format(k+1)))
                
                frames_k = frames[k].to(device)
                image_mse_loss = mse_loss(pred_images, frames_k)
                temp_train_mse_loss.append(image_mse_loss.data.item())
                loss += image_mse_loss
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch {} batch {}/{} training done with average BCE loss={} and MSE loss={}.".format(i+1, j+1, len(train_dataloader), sum(temp_train_bce_loss) / len(temp_train_bce_loss), sum(temp_train_mse_loss) / len(temp_train_mse_loss)))

        print("Epoch {} train BCE loss={} and MSE loss={}".format(i+1, sum(temp_train_bce_loss) / len(temp_train_bce_loss), sum(temp_train_mse_loss) / len(temp_train_mse_loss)))
        train_bce_loss_per_epoch.append(sum(temp_train_bce_loss) / len(temp_train_bce_loss))
        train_mse_loss_per_epoch.append(sum(temp_train_mse_loss) / len(temp_train_mse_loss))

    with open('loss.txt', 'w') as f:
        f.write(str(train_bce_loss_per_epoch))
        f.write(str(train_mse_loss_per_epoch))


if __name__ == '__main__':
    num_epoch = 10
    batch_size = 2
    teacher_forcing_prob = 0.5
    first_n_frame_dynamics = 5
    frame_interval = 2

    assert teacher_forcing_prob >= 0 and teacher_forcing_prob <= 1
    assert frame_interval > 0 and type(frame_interval) == int

    main('contact', num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval)