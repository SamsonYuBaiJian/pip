import torch.nn as nn
import torch
from dataloader import Data
from model import Model
from torch.utils.data import DataLoader
import numpy as np


def main(task, num_epoch, batch_size):
    train_dataset = Data('/mnt/c/Users/samso/Documents/SamsonYuBaiJian/CLEVEREST/dataset/contact/labels.csv', '/mnt/c/Users/samso/Documents/SamsonYuBaiJian/CLEVEREST/dataset/contact/frames')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = Model()
    # turn off gradients for other tasks
    tasks = ['contact', 'contain', 'stability']
    for i in tasks:
        if i != task:
            for n, p in model.named_parameters():
                if i in n:
                    p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    bce_logits_loss = nn.BCEWithLogitsLoss()
    # TODO: add MSE loss for generated images
    # mse_loss = nn.MSELoss()

    model.train()
    print('Training...')
    train_loss_per_epoch = []
    for i in range(num_epoch):
        temp_train_loss = []
        for j, batch in enumerate(train_dataloader):
            frames, coordinates, labels = batch
            coordinates = torch.stack(coordinates).T
            preds = model(task, coordinates, frames)
            labels = torch.unsqueeze(labels, dim=1).type_as(preds)
            loss = bce_logits_loss(preds, labels) # + mse_loss()
            temp_train_loss.append(loss.data.item())
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch {} batch {}/{} training done with average loss = {}.".format(i+1, j+1, len(train_dataloader), sum(temp_train_loss) / len(temp_train_loss)))

        print("Epoch {} train loss =".format(i+1), sum(temp_train_loss) / len(temp_train_loss))
        train_loss_per_epoch.append(sum(temp_train_loss) / len(temp_train_loss))

    with open('loss.txt', 'w') as f:
        f.write(str(train_loss_per_epoch))


if __name__ == '__main__':
    num_epoch = 10
    batch_size = 6

    main('contact', num_epoch, batch_size)