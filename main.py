import torch.nn as nn
import torch
from dataloader import Data
from model import Model
from torch.utils.data import DataLoader


def main(task, num_epoch, batch_size):
    train_dataset = Data('/mnt/c/Users/samso/Documents/SamsonYuBaiJian/CLEVEREST/dataset/contact/labels.csv', '/mnt/c/Users/samso/Documents/SamsonYuBaiJian/CLEVEREST/dataset/contact/frames')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = Model()
    bce_logits_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(num_epoch):
        for batch in train_dataloader:
            frames, coordinates, labels = batch['frames'], batch['coordinates'], batch['labels']
            print(len(frames), coordinates, labels)
            preds = model(task, coordinates, frames)
            loss = bce_logits_loss(preds, labels) # + mse_loss()
            # graftnet_classify_train_loss_epoch.append(graftnet_loss.data.item())
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(graftnet.parameters(), gradient_clip)
            optimizer.step()


if __name__ == '__main__':
    num_epoch = 10
    batch_size = 4

    main('contact', num_epoch, batch_size)