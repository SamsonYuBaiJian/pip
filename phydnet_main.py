import torch.nn as nn
import torch
from dataloader import Data
from model import Model
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import random
import yaml
import argparse
import datetime
import os
import math
import json
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from phydnet_model import ConvLSTM, PhyCell, EncoderRNN, PhyDNet
from constrain_moments import K2M


def get_classification_accuracy(pred_labels, labels):
    """
    Get accuracy for classification.
    """
    size = pred_labels.shape[0]
    mask = pred_labels >= 0
    num_correct = torch.sum(mask == labels).item()
    acc = num_correct / size

    return acc, num_correct


def phydnet_train(cfg, task_type, frame_path, mask_path, train_label_path, val_label_path, test_label_path, save_path, load_model_path, num_epoch, batch_size, first_n_frame_dynamics, frame_interval, save_spans, max_seq_len, device):
    # get experiment ID
    model_type = 'phydnet'
    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +  ' ' + task_type + ' train' + ' ' + model_type
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    experiment_save_path = os.path.join(save_path, experiment_id)
    os.makedirs(experiment_save_path, exist_ok=True)
    
    if task_type == 'combined':
        train_combine_idx = {'contact': [1,200], 'contain': [201,400], 'stability': [401,600]}
        train_dataset = Data(frame_path, mask_path, train_label_path, frame_interval, first_n_frame_dynamics, task_type, model_type, max_seq_len, combined_scene_tasks=train_combine_idx)
    else:
        train_dataset = Data(frame_path, mask_path, train_label_path, frame_interval, first_n_frame_dynamics, task_type, model_type, max_seq_len,)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    if task_type == 'combined':
        val_combine_idx = {'contact': [601,666], 'contain': [667,733], 'stability': [734,800]}
        val_dataset = Data(frame_path, mask_path, val_label_path, frame_interval, first_n_frame_dynamics, task_type, model_type, max_seq_len, combined_scene_tasks=val_combine_idx)
    else:
        val_dataset = Data(frame_path, mask_path, val_label_path, frame_interval, first_n_frame_dynamics, task_type, model_type, max_seq_len,)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    if load_model_path is not None:
        model = torch.load(load_model_path, map_location=device).to(device)
        model.device = device
        model.encoder.phycell.device = device
        model.encoder.convcell.device = device
        model.to(device)
    else:
        model = PhyDNet(device)
        model.to(device)

    # constraints for training
    constraints = torch.zeros((49,7,7)).to(device)
    ind = 0
    for i in range(0,7):
        for j in range(0,7):
            constraints[ind,i,j] = 1
            ind +=1
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2,factor=0.1,verbose=True)
    bce_logits_loss = nn.BCEWithLogitsLoss().to(device)
    frame_loss = nn.MSELoss().to(device)

    stats = {'train': {'cls_loss': [], 'cls_acc': [], 'gen_loss': []}, 'val': {'cls_loss': [], 'cls_acc': [], 'gen_loss': []}}
    
    max_val_classification_acc = 0
    max_val_classification_epoch = None
    min_val_image_loss = 100
    min_val_image_epoch = None

    for i in range(num_epoch):
        # training
        print('Training for epoch {}/{}...'.format(i+1, num_epoch))
        teacher_forcing_ratio = np.maximum(0 , 1 - i * 0.003) 
        temp_train_classification_loss = []
        temp_train_image_loss = []
        total_num_correct = 0
        total_cnt = 0
        model.train()

        for j, batch in tqdm(enumerate(train_dataloader)):
            frames, masks, labels, queries = batch
            retrieved_batch_size = len(frames[0])
            total_cnt += retrieved_batch_size
            # teacher_forcing_batch = random.choices(population=[True, False], weights=[teacher_forcing_prob, 1-teacher_forcing_prob], k=retrieved_batch_size)
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
            pred_labels, pred_images_seq, decoded_first_n_frames = model(task_type, frames, masks, queries, use_teacher_forcing, first_n_frame_dynamics, max_seq_len)
            labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
            train_acc, num_correct = get_classification_accuracy(pred_labels, labels)
            total_num_correct += num_correct
            cls_loss = bce_logits_loss(pred_labels, labels)
            loss = cls_loss

            temp_train_classification_loss.append(cls_loss.data.item() * retrieved_batch_size)
            temp_train_image_loss.append(0)

            # add input sequence decoded images loss
            for k, pred_images in enumerate(decoded_first_n_frames):
                frames_k = frames[k+1].to(device)
                pred_images = torch.clamp(pred_images, 0, 1)
                img_loss = frame_loss(pred_images, frames_k)
                loss += img_loss
                temp_train_image_loss[-1] += img_loss.data.item()

            for k, pred_images in enumerate(pred_images_seq):
                frames_k = frames[k+first_n_frame_dynamics].to(device)
                pred_images = torch.clamp(pred_images, 0, 1)
                img_loss = frame_loss(pred_images, frames_k)
                loss += img_loss
                temp_train_image_loss[-1] += img_loss.data.item()
                seq_len = len(pred_images_seq)
            temp_train_image_loss[-1] /= seq_len

            # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
            k2m = K2M([7,7]).to(device)
            for b in range(0, model.phycell.cell_list[0].input_dim):
                filters = model.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
                m = k2m(filters.double()) 
                m  = m.float()   
                loss += frame_loss(m, constraints) # constrains is a precomputed matrix

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch {}/{} batch {}/{} training done with cls loss={}, cls accuracy={}, gen loss={}.".format(i+1, num_epoch, j+1, len(train_dataloader), temp_train_classification_loss[-1] / retrieved_batch_size, train_acc, temp_train_image_loss[-1] / retrieved_batch_size))

        if (i+1) % 10 == 0:
            mse = evaluate(model, val_dataloader, task_type, frames, masks, queries, False, first_n_frame_dynamics, max_seq_len, device) 
            scheduler.step(mse)                   
            # torch.save(model.state_dict(),'save/encoder_{}.pth'.format(name)) 

        print("\nEpoch {}/{} OVERALL train cls loss={}, cls accuracy={}, gen loss={}.\n".format(i+1, num_epoch, sum(temp_train_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_train_image_loss)))
        stats['train']['cls_loss'].append(sum(temp_train_classification_loss) / total_cnt)
        stats['train']['cls_acc'].append(total_num_correct / total_cnt)
        stats['train']['gen_loss'].append(sum(temp_train_image_loss))

        # save frames
        epoch_train_save_dir = os.path.join(os.path.join(experiment_save_path, 'epoch_{}'.format(str(i+1))), 'train')
        gen_save_img_dir = os.path.join(epoch_train_save_dir, 'gen')
        real_save_img_dir = os.path.join(epoch_train_save_dir, 'real')
        os.makedirs(gen_save_img_dir, exist_ok=True)
        os.makedirs(real_save_img_dir, exist_ok=True)
        for k in range(first_n_frame_dynamics):
            save_image(frames[k][0], os.path.join(gen_save_img_dir, '{}.png'.format(k+1)))
            save_image(frames[k][0], os.path.join(real_save_img_dir, '{}.png'.format(k+1)))
            save_image(masks[k][0], os.path.join(gen_save_img_dir, '{}_mask.png'.format(k+1)))
            save_image(masks[k][0], os.path.join(real_save_img_dir, '{}_mask.png'.format(k+1)))
        for k, pred_images in enumerate(pred_images_seq):
            save_image(pred_images[0], os.path.join(gen_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
            save_image(frames[k+first_n_frame_dynamics][0], os.path.join(real_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))

        # validation
        print('Validation for epoch {}/{}...'.format(i+1, num_epoch))
        temp_val_classification_loss = []
        temp_val_image_loss = []
        all_span_list = []
        temp_val_image_loss = []
        total_num_correct = 0
        total_cnt = 0
        model.eval()
        with torch.no_grad():
            for j, batch in tqdm(enumerate(val_dataloader)):
                frames, masks, labels, queries = batch
                retrieved_batch_size = len(frames[0])
                total_cnt += retrieved_batch_size
                # no teacher forcing for validation
                use_teacher_forcing = False
                # teacher_forcing_batch = random.choices(population=[True, False], weights=[0, 1], k=retrieved_batch_size)
                pred_labels, pred_images_seq, decoded_first_n_frames = model(task_type, frames, masks, queries, use_teacher_forcing, first_n_frame_dynamics, max_seq_len)
                labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
                val_acc, num_correct = get_classification_accuracy(pred_labels, labels)
                total_num_correct += num_correct
                bce_loss = bce_logits_loss(pred_labels, labels)
                temp_val_classification_loss.append(bce_loss.data.item() * retrieved_batch_size)

                temp_val_image_loss.append(0)

                # add input sequence decoded images loss
                for k, pred_images in enumerate(decoded_first_n_frames):
                    frames_k = frames[k+1].to(device)
                    pred_images = torch.clamp(pred_images, 0, 1)
                    img_loss = frame_loss(pred_images, frames_k)
                    loss += img_loss
                    temp_val_image_loss[-1] += img_loss.data.item()

                # temp_val_jsd_loss.append(torch.mean(jsd_loss).data.item() * retrieved_batch_size)
                for k, pred_images in enumerate(pred_images_seq):
                    frames_k = frames[k+first_n_frame_dynamics].to(device)
                    pred_images = torch.clamp(pred_images, 0, 1)
                    img_loss = frame_loss(pred_images, frames_k)
                    temp_val_image_loss[-1] += img_loss.data.item() * retrieved_batch_size
                    seq_len = len(pred_images_seq)
                temp_val_image_loss[-1] /= seq_len
                
                print("Epoch {}/{} batch {}/{} validation done with cls loss={}, cls accuracy={}, gen loss={}.".format(i+1, num_epoch, j+1, len(val_dataloader), temp_val_classification_loss[-1] / retrieved_batch_size, val_acc, temp_val_image_loss[-1] / retrieved_batch_size))

        print("\nEpoch {}/{} OVERALL validation cls loss={}, cls accuracy={}, gen loss={}.\n".format(i+1, num_epoch, sum(temp_val_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_val_image_loss)))
        stats['val']['cls_loss'].append(sum(temp_val_classification_loss) / total_cnt)
        stats['val']['cls_acc'].append(total_num_correct / total_cnt)
        stats['val']['gen_loss'].append(sum(temp_val_image_loss))

        # save frames and spans
        epoch_val_save_dir = os.path.join(os.path.join(experiment_save_path, 'epoch_{}'.format(str(i+1))), 'val')
        gen_save_img_dir = os.path.join(epoch_val_save_dir, 'gen')
        real_save_img_dir = os.path.join(epoch_val_save_dir, 'real')
        os.makedirs(gen_save_img_dir, exist_ok=True)
        os.makedirs(real_save_img_dir, exist_ok=True)
        for k in range(first_n_frame_dynamics):
            save_image(frames[k][0], os.path.join(gen_save_img_dir, '{}.png'.format(k+1)))
            save_image(frames[k][0], os.path.join(real_save_img_dir, '{}.png'.format(k+1)))
            save_image(masks[k][0], os.path.join(gen_save_img_dir, '{}_mask.png'.format(k+1)))
            save_image(masks[k][0], os.path.join(real_save_img_dir, '{}_mask.png'.format(k+1)))
        for k, pred_images in enumerate(pred_images_seq):
            save_image(pred_images[0], os.path.join(gen_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
            save_image(frames[k+first_n_frame_dynamics][0], os.path.join(real_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))

        print("Saved new validation frames.\n")


        # check for best stat/model using validation accuracy
        if stats['val']['cls_acc'][-1] >= max_val_classification_acc:
            max_val_classification_acc = stats['val']['cls_acc'][-1]
            max_val_classification_epoch = i+1
            torch.save(model, os.path.join(experiment_save_path, 'model'))
        if stats['val']['gen_loss'][-1] < min_val_image_loss:
            min_val_image_loss = stats['val']['gen_loss'][-1]
            min_val_image_epoch = i+1

        with open(os.path.join(experiment_save_path, 'log.txt'), 'w') as f:
            f.write('{}\n'.format(cfg))
            f.write('{}\n'.format(stats))
            f.write('Max val classification acc: epoch {}, {}\n'.format(max_val_classification_epoch, max_val_classification_acc))
            f.write('Min val generation loss: epoch {}, {}\n'.format(min_val_image_epoch, min_val_image_loss))
            f.close()



def evaluate(model, loader, task_type, frames, masks, queries, use_teacher_forcing, first_n_frame_dynamics, max_seq_len, device):
    total_mse = 0
    with torch.no_grad():
        for j, batch in tqdm(enumerate(loader)):
            frames, masks, labels, queries = batch
            retrieved_batch_size = len(frames[0])

            _, pred_images_seq, _ = model(task_type, frames, masks, queries, use_teacher_forcing, first_n_frame_dynamics, max_seq_len)
            all_decoded_frames = torch.stack(frames[first_n_frame_dynamics:]).swapaxes(0,1).cpu().numpy()
            all_pred_images_seq = torch.stack(pred_images_seq).swapaxes(0,1).cpu().numpy()

            mse_batch = np.mean((all_pred_images_seq-all_decoded_frames)**2, axis=(0,1,2)).sum()
            total_mse += mse_batch
     
    print('eval mse ', total_mse/len(loader))
     
    return total_mse/len(loader)


def phydnet_test(cfg, task_type, frame_path, mask_path, train_label_path, val_label_path, test_label_path, save_path, load_model_path, num_epoch, batch_size, first_n_frame_dynamics, frame_interval, save_spans, max_seq_len, device):
    # get experiment ID
    model_type = 'phydnet'
    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +  ' ' + task_type + ' test' + ' ' + model_type
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    experiment_save_path = os.path.join(save_path, experiment_id)
    os.makedirs(experiment_save_path, exist_ok=True)

    if task_type == 'combined':
        test_combine_idx = {'contact': [801,866], 'contain': [867,933], 'stability': [934,1000]}
        test_dataset = Data(frame_path, mask_path, test_label_path, frame_interval, first_n_frame_dynamics, task_type, model_type, max_seq_len, combined_scene_tasks=test_combine_idx)
    else:
        test_dataset = Data(frame_path, mask_path, test_label_path, frame_interval, first_n_frame_dynamics, task_type, model_type, max_seq_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # NOTE
    save_img_dir = os.path.join(experiment_save_path, 'generations')

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    if load_model_path is not None:
        model = torch.load(load_model_path, map_location=device).to(device)
        model.device = device
        model.encoder.phycell.device = device
        model.encoder.convcell.device = device
        model.to(device)

    else:
        model = PhyDNet(device)
        model.to(device)
        constraints = torch.zeros((49,7,7)).to(device)
        ind = 0
        for i in range(0,7):
            for j in range(0,7):
                constraints[ind,i,j] = 1
                ind +=1

    bce_logits_loss = nn.BCEWithLogitsLoss().to(device)
    frame_loss = nn.MSELoss().to(device)

    stats = {'test': {'cls_loss': [], 'cls_acc': [], 'gen_loss': []}}

    print('Testing...')
    model.eval()
    temp_test_classification_loss = []
    temp_test_image_loss = []
    temp_test_image_loss = []
    total_num_correct = 0
    total_cnt = 0

    with torch.no_grad():
        for j, batch in tqdm(enumerate(test_dataloader)):
            frames, masks, labels, queries = batch
            retrieved_batch_size = len(frames[0])
            total_cnt += retrieved_batch_size
            use_teacher_forcing = False
            pred_labels, pred_images_seq, decoded_first_n_frames = model(task_type, frames, masks, queries, use_teacher_forcing, first_n_frame_dynamics, max_seq_len)
            labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
            test_acc, num_correct = get_classification_accuracy(pred_labels, labels)
            total_num_correct += num_correct
            cls_loss = bce_logits_loss(pred_labels, labels)
            temp_test_classification_loss.append(cls_loss.data.item() * retrieved_batch_size)

            temp_test_image_loss.append(0)

            # add input sequence decoded images loss
            for k, pred_images in enumerate(decoded_first_n_frames):
                frames_k = frames[k+1].to(device)
                pred_images = torch.clamp(pred_images, 0, 1)
                img_loss = frame_loss(pred_images, frames_k)
                temp_test_image_loss[-1] += img_loss.data.item()

            for k, pred_images in enumerate(pred_images_seq):
                frames_k = frames[k+first_n_frame_dynamics].to(device)
                pred_images = torch.clamp(pred_images, 0, 1)
                img_loss = frame_loss(pred_images, frames_k)
                temp_test_image_loss[-1] += img_loss.data.item()
                seq_len = len(pred_images_seq)
            temp_test_image_loss[-1] /= seq_len

            print("Batch {}/{} testing done with cls loss={}, cls accuracy={}, gen loss={}.".format(j+1, len(test_dataloader), temp_test_classification_loss[-1] / retrieved_batch_size, test_acc, temp_test_image_loss[-1] / retrieved_batch_size))

        print("OVERALL test cls loss={}, cls accuracy={}, gen loss={}.\n".format(sum(temp_test_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_test_image_loss)))
        stats['test']['cls_loss'].append(sum(temp_test_classification_loss) / total_cnt)
        stats['test']['cls_acc'].append(total_num_correct / total_cnt)
        stats['test']['gen_loss'].append(sum(temp_test_image_loss))

        # save frames and spans
        gen_save_img_dir = os.path.join(experiment_save_path, 'gen')
        real_save_img_dir = os.path.join(experiment_save_path, 'real')
        os.makedirs(gen_save_img_dir, exist_ok=True)
        os.makedirs(real_save_img_dir, exist_ok=True)
        for k in range(first_n_frame_dynamics):
            save_image(frames[k][0], os.path.join(gen_save_img_dir, '{}.png'.format(k+1)))
            save_image(frames[k][0], os.path.join(real_save_img_dir, '{}.png'.format(k+1)))
            save_image(masks[k][0], os.path.join(gen_save_img_dir, '{}_mask.png'.format(k+1)))
            save_image(masks[k][0], os.path.join(real_save_img_dir, '{}_mask.png'.format(k+1)))
        for k, pred_images in enumerate(pred_images_seq):
            save_image(pred_images[0], os.path.join(gen_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))
            save_image(frames[k+first_n_frame_dynamics][0], os.path.join(real_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics+1)))

        print("Saved new test frames.\n")

        with open(os.path.join(experiment_save_path, 'log.txt'), 'w') as f:
            f.write('{}\n'.format(cfg))
            f.write('{}\n'.format(stats))
            f.close()