import torch.nn as nn
import torch
from dataloader import Data
from model import Model
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import random
from piqa import PSNR
import yaml
import argparse
import datetime
import os
import math
import json
from phydnet_main import phydnet_train, phydnet_test


def get_classification_accuracy(pred_labels, labels):
    """
    Get accuracy for classification.
    """
    size = pred_labels.shape[0]
    mask = pred_labels >= 0
    num_correct = torch.sum(mask == labels).item()
    acc = num_correct / size

    return acc, num_correct



def get_span_threshold(seq_len):
    forward_probs = [0] * seq_len
    backward_probs = [0] * seq_len
    span_threshold = [0] * seq_len

    for i in range(seq_len):
        interval = int(i+1) / seq_len
        forward_probs[i] = interval
        backward_probs[-i-1] = interval

    for i in range(seq_len):
        span_threshold[i] = forward_probs[i] * backward_probs[i]
    span_sum = sum(span_threshold)

    for i in range(seq_len):
        span_threshold[i] /= span_sum

    return span_threshold



def train(cfg, task_type, frame_path, mask_path, train_label_path, val_label_path, test_label_path, save_path, load_model_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, learning_rate, save_spans, max_seq_len, span_num, device, model_type):
    # get experiment ID
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
        if model_type == 'pip':
            model.ConvLSTMCell1.device = device
            model.ConvLSTMCell2.device = device
            model.ConvLSTMCell3.device = device
            model.span_predict.device = device
        elif model_type == 'ablation':
            model.ConvLSTMCell1.device = device
            model.ConvLSTMCell2.device = device
            model.ConvLSTMCell3.device = device
            model.ablation.device = device
        elif model_type == 'baseline':
            model.baseline.device = device
    else:
        model = Model(device, span_num, model_type, nc=3, nf=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bce_logits_loss = nn.BCEWithLogitsLoss().to(device)
    if model_type == 'pip' or model_type == 'ablation':
        frame_loss = PSNR().to(device)

    if model_type == 'pip' or model_type == 'ablation':
        stats = {'train': {'cls_loss': [], 'cls_acc': [], 'gen_loss': []}, 'val': {'cls_loss': [], 'cls_acc': [], 'gen_loss': []}}
    else:
        stats = {'train': {'cls_loss': [], 'cls_acc': []}, 'val': {'cls_loss': [], 'cls_acc': []}}
    
    max_val_classification_acc = 0
    max_val_classification_epoch = None
    if model_type == 'pip' or model_type == 'ablation':
        max_val_image_loss = 0
        max_val_image_epoch = None

    for i in range(num_epoch):
        # training
        print('Training for epoch {}/{}...'.format(i+1, num_epoch))
        temp_train_classification_loss = []
        if model_type == 'pip':
            temp_train_image_loss = []
            temp_train_jsd_loss = []
            all_span_list = []
        elif model_type == 'ablation':
            temp_train_image_loss = []
        total_num_correct = 0
        total_cnt = 0
        model.train()
        for j, batch in tqdm(enumerate(train_dataloader)):
            frames, masks, labels, queries = batch
            retrieved_batch_size = len(frames[0])
            total_cnt += retrieved_batch_size
            teacher_forcing_batch = random.choices(population=[True, False], weights=[teacher_forcing_prob, 1-teacher_forcing_prob], k=retrieved_batch_size)
            pred_labels, pred_images_seq, all_r, jsd_loss = model(task_type, frames, masks, queries, teacher_forcing_batch, first_n_frame_dynamics, max_seq_len)
            labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
            train_acc, num_correct = get_classification_accuracy(pred_labels, labels)
            total_num_correct += num_correct
            cls_loss = bce_logits_loss(pred_labels, labels)
            if model_type == 'pip':
                loss = cls_loss + torch.mean(jsd_loss)
            elif model_type == 'ablation' or model_type == 'baseline':
                loss = cls_loss

            temp_train_classification_loss.append(cls_loss.data.item() * retrieved_batch_size)
            if model_type == 'pip' or model_type == 'ablation':
                temp_train_image_loss.append(0)
                if model_type == 'pip':
                    temp_train_jsd_loss.append(torch.mean(jsd_loss).data.item() * retrieved_batch_size)
                for k, pred_images in enumerate(pred_images_seq):
                    frames_k = frames[k+first_n_frame_dynamics].to(device)
                    pred_images = torch.clamp(pred_images, 0, 1)
                    img_loss = frame_loss(pred_images, frames_k)
                    loss += -img_loss
                    temp_train_image_loss[-1] += img_loss.data.item() * retrieved_batch_size
                    seq_len = len(pred_images_seq)
                temp_train_image_loss[-1] /= seq_len

            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if save_spans:
                span_threshold = get_span_threshold(seq_len)
                for batch_spans in all_r:
                    batch_span_indices = []
                    for k, span in enumerate(batch_spans):
                        span_indices = []
                        for l, frame_score in enumerate(span):
                            if frame_score.item() > span_threshold[l]:
                                span_indices.append(l+first_n_frame_dynamics+1)
                        batch_span_indices.append(span_indices)
                    all_span_list.append(batch_span_indices)

            if model_type == 'pip':
                print("Epoch {}/{} batch {}/{} training done with cls loss={}, cls accuracy={}, gen loss={}, jsd loss={}.".format(i+1, num_epoch, j+1, len(train_dataloader), temp_train_classification_loss[-1] / retrieved_batch_size, train_acc, temp_train_image_loss[-1] / retrieved_batch_size, temp_train_jsd_loss[-1] / retrieved_batch_size))
            elif model_type == 'ablation':
                print("Epoch {}/{} batch {}/{} training done with cls loss={}, cls accuracy={}, gen loss={}.".format(i+1, num_epoch, j+1, len(train_dataloader), temp_train_classification_loss[-1] / retrieved_batch_size, train_acc, temp_train_image_loss[-1] / retrieved_batch_size))
            elif model_type == 'baseline':
                print("Epoch {}/{} batch {}/{} training done with cls loss={}, cls accuracy={}.".format(i+1, num_epoch, j+1, len(train_dataloader), temp_train_classification_loss[-1] / retrieved_batch_size, train_acc))

        if model_type == 'pip':
            print("\nEpoch {}/{} OVERALL train cls loss={}, cls accuracy={}, gen loss={}, jsd loss={}.\n".format(i+1, num_epoch, sum(temp_train_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_train_image_loss) / total_cnt, sum(temp_train_jsd_loss) / total_cnt))
        elif model_type == 'ablation':
            print("\nEpoch {}/{} OVERALL train cls loss={}, cls accuracy={}, gen loss={}.\n".format(i+1, num_epoch, sum(temp_train_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_train_image_loss) / total_cnt))
        elif model_type == 'baseline':
            print("Epoch {}/{} OVERALL train cls loss={}, cls accuracy={}.\n".format(i+1, num_epoch, sum(temp_train_classification_loss) / total_cnt, total_num_correct / total_cnt))
        stats['train']['cls_loss'].append(sum(temp_train_classification_loss) / total_cnt)
        stats['train']['cls_acc'].append(total_num_correct / total_cnt)
        if model_type == 'pip' or model_type == 'ablation':
            stats['train']['gen_loss'].append(sum(temp_train_image_loss) / total_cnt)

        # save frames and spans
        if model_type == 'pip' or model_type == 'ablation':
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
            
            if model_type == 'pip':
                span_threshold = get_span_threshold(seq_len)
                with open(os.path.join(epoch_train_save_dir, 'spans.txt'), 'w') as f:
                    for k, span in enumerate(all_r[0]):
                        span_indices = []
                        for l, frame_score in enumerate(span):
                            if frame_score.item() > span_threshold[l]:
                                span_indices.append(l+first_n_frame_dynamics+1)
                        f.write('Span {}: '.format(k+1) + str(span_indices) + '\n')

                if teacher_forcing_batch[0]:
                    print("Saved new train frames and spans WITH teacher forcing.\n")
                else:
                    print("Saved new train frames and spans WITHOUT teacher forcing.\n")
            elif model_type == 'ablation':
                if teacher_forcing_batch[0]:
                    print("Saved new train frames WITH teacher forcing.\n")
                else:
                    print("Saved new train frames WITHOUT teacher forcing.\n")

            if save_spans and model_type == 'pip':
                with open(os.path.join(epoch_train_save_dir, 'all_spans.json'), 'w') as f:
                    json.dump(all_span_list, f)
                    f.close()
                print("Saved all selected spans.\n")



        # validation
        print('Validation for epoch {}/{}...'.format(i+1, num_epoch))
        temp_val_classification_loss = []
        if model_type == 'pip':
            temp_val_image_loss = []
            temp_val_jsd_loss = []
            all_span_list = []
        elif model_type == 'ablation':
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
                teacher_forcing_batch = random.choices(population=[True, False], weights=[0, 1], k=retrieved_batch_size)
                pred_labels, pred_images_seq, all_r, jsd_loss = model(task_type, frames, masks, queries, teacher_forcing_batch, first_n_frame_dynamics, max_seq_len)
                labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
                val_acc, num_correct = get_classification_accuracy(pred_labels, labels)
                total_num_correct += num_correct
                bce_loss = bce_logits_loss(pred_labels, labels)
                temp_val_classification_loss.append(bce_loss.data.item() * retrieved_batch_size)
                
                if model_type == 'pip' or model_type == 'ablation':
                    temp_val_image_loss.append(0)
                    if model_type == 'pip':
                        temp_val_jsd_loss.append(torch.mean(jsd_loss).data.item() * retrieved_batch_size)
                    for k, pred_images in enumerate(pred_images_seq):
                        frames_k = frames[k+first_n_frame_dynamics].to(device)
                        pred_images = torch.clamp(pred_images, 0, 1)
                        img_loss = frame_loss(pred_images, frames_k)
                        temp_val_image_loss[-1] += img_loss.data.item() * retrieved_batch_size
                        seq_len = len(pred_images_seq)
                    temp_val_image_loss[-1] /= seq_len

                if save_spans and model_type == 'pip':
                    span_threshold = get_span_threshold(seq_len)
                    for batch_spans in all_r:
                        batch_span_indices = []
                        for k, span in enumerate(batch_spans):
                            span_indices = []
                            for l, frame_score in enumerate(span):
                                if frame_score.item() > span_threshold[l]:
                                    span_indices.append(l+first_n_frame_dynamics+1)
                            batch_span_indices.append(span_indices)
                        all_span_list.append(batch_span_indices)
                
                if model_type == 'pip':
                    print("Epoch {}/{} batch {}/{} validation done with cls loss={}, cls accuracy={}, gen loss={}, jsd loss={}.".format(i+1, num_epoch, j+1, len(val_dataloader), temp_val_classification_loss[-1] / retrieved_batch_size, val_acc, temp_val_image_loss[-1] / retrieved_batch_size, temp_val_jsd_loss[-1] / retrieved_batch_size))
                elif model_type == 'ablation':
                    print("Epoch {}/{} batch {}/{} validation done with cls loss={}, cls accuracy={}, gen loss={}.".format(i+1, num_epoch, j+1, len(val_dataloader), temp_val_classification_loss[-1] / retrieved_batch_size, val_acc, temp_val_image_loss[-1] / retrieved_batch_size))
                elif model_type == 'baseline':
                    print("Epoch {}/{} batch {}/{} validation done with cls loss={}, cls accuracy={}.".format(i+1, num_epoch, j+1, len(val_dataloader), temp_val_classification_loss[-1] / retrieved_batch_size, val_acc))

        if model_type == 'pip' or model_type == 'ablation':
            print("\nEpoch {}/{} OVERALL validation cls loss={}, cls accuracy={}, gen loss={}, jsd loss={}.\n".format(i+1, num_epoch, sum(temp_val_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_val_image_loss) / total_cnt, sum(temp_val_jsd_loss) / total_cnt))
        elif model_type == 'ablation':
            print("\nEpoch {}/{} OVERALL validation cls loss={}, cls accuracy={}, gen loss={}.\n".format(i+1, num_epoch, sum(temp_val_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_val_image_loss) / total_cnt))
        elif model_type == 'baseline':
            print("Epoch {}/{} OVERALL validation cls loss={}, cls accuracy={}.\n".format(i+1, num_epoch, sum(temp_val_classification_loss) / total_cnt, total_num_correct / total_cnt))
        stats['val']['cls_loss'].append(sum(temp_val_classification_loss) / total_cnt)
        stats['val']['cls_acc'].append(total_num_correct / total_cnt)
        if model_type == 'pip' or model_type == 'ablation':
            stats['val']['gen_loss'].append(sum(temp_val_image_loss) / total_cnt)

        # save frames and spans
        if model_type == 'pip' or model_type == 'ablation':
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
            
            if model_type == 'pip':
                span_threshold = get_span_threshold(seq_len)
                with open(os.path.join(epoch_val_save_dir, 'spans.txt'), 'w') as f:
                    for k, span in enumerate(all_r[0]):
                        span_indices = []
                        for l, frame_score in enumerate(span):
                            if frame_score.item() > span_threshold[l]:
                                span_indices.append(l+first_n_frame_dynamics+1)
                        f.write('Span {}: '.format(k+1) + str(span_indices) + '\n')

                print("Saved new validation frames and spans.\n")
            elif model_type == 'ablation':
                print("Saved new validation frames.\n")

            if save_spans and model_type == 'pip':
                with open(os.path.join(epoch_val_save_dir, 'all_spans.json'), 'w') as f:
                    json.dump(all_span_list, f)
                    f.close()
                print("Saved all selected spans.\n")

        # check for best stat/model using validation accuracy
        if stats['val']['cls_acc'][-1] >= max_val_classification_acc:
            max_val_classification_acc = stats['val']['cls_acc'][-1]
            max_val_classification_epoch = i+1
            torch.save(model, os.path.join(experiment_save_path, 'model'))
        if model_type == 'pip' or model_type == 'ablation':
            if stats['val']['gen_loss'][-1] > max_val_image_loss:
                max_val_image_loss = stats['val']['gen_loss'][-1]
                max_val_image_epoch = i+1

        with open(os.path.join(experiment_save_path, 'log.txt'), 'w') as f:
            f.write('{}\n'.format(cfg))
            f.write('{}\n'.format(stats))
            f.write('Max val classification acc: epoch {}, {}\n'.format(max_val_classification_epoch, max_val_classification_acc))
            if model_type == 'pip' or model_type == 'ablation':
                f.write('Max val generation loss: epoch {}, {}\n'.format(max_val_image_epoch, max_val_image_loss))
            f.close()



def test(cfg, task_type, frame_path, mask_path, train_label_path, val_label_path, test_label_path, save_path, load_model_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, learning_rate, save_spans, max_seq_len, span_num, device, model_type):
    # get experiment ID
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
        if model_type == 'pip':
            model.ConvLSTMCell1.device = device
            model.ConvLSTMCell2.device = device
            model.ConvLSTMCell3.device = device
            model.span_predict.device = device
        elif model_type == 'ablation':
            model.ConvLSTMCell1.device = device
            model.ConvLSTMCell2.device = device
            model.ConvLSTMCell3.device = device
            model.ablation.device = device
        elif model_type == 'baseline':
            model.baseline.device = device
    else:
        model = Model(device, span_num, model_type, nc=3, nf=16).to(device)
    bce_logits_loss = nn.BCEWithLogitsLoss().to(device)
    if model_type == 'pip' or model_type == 'ablation':
        frame_loss = PSNR().to(device)

    if model_type == 'pip' or model_type == 'ablation':
        stats = {'test': {'cls_loss': [], 'cls_acc': [], 'gen_loss': []}}
    elif model_type == 'baseline':
        stats = {'test': {'cls_loss': [], 'cls_acc': []}}

    print('Testing...')
    model.eval()
    temp_test_classification_loss = []
    if model_type == 'pip':
        temp_test_image_loss = []
        temp_test_jsd_loss = []
        all_span_list = []
    elif model_type == 'ablation':
        temp_test_image_loss = []
    total_num_correct = 0
    total_cnt = 0

    with torch.no_grad():
        for j, batch in tqdm(enumerate(test_dataloader)):
            frames, masks, labels, queries = batch
            retrieved_batch_size = len(frames[0])
            total_cnt += retrieved_batch_size
            teacher_forcing_batch = random.choices(population=[True, False], weights=[0, 1], k=retrieved_batch_size)
            pred_labels, pred_images_seq, all_r, jsd_loss = model(task_type, frames, masks, queries, teacher_forcing_batch, first_n_frame_dynamics, max_seq_len)
            labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels)
            test_acc, num_correct = get_classification_accuracy(pred_labels, labels)
            total_num_correct += num_correct
            cls_loss = bce_logits_loss(pred_labels, labels)

            temp_test_classification_loss.append(cls_loss.data.item() * retrieved_batch_size)
            if model_type == 'pip' or model_type == 'ablation':
                temp_test_image_loss.append(0)
                if model_type == 'pip':
                    temp_test_jsd_loss.append(torch.mean(jsd_loss).data.item() * retrieved_batch_size)
                for k, pred_images in enumerate(pred_images_seq):
                    frames_k = frames[k+first_n_frame_dynamics].to(device)
                    pred_images = torch.clamp(pred_images, 0, 1)
                    img_loss = frame_loss(pred_images, frames_k)
                    # loss += -img_loss
                    temp_test_image_loss[-1] += img_loss.data.item() * retrieved_batch_size
                    seq_len = len(pred_images_seq)
                temp_test_image_loss[-1] /= seq_len

            if save_spans and model_type == 'pip':
                span_threshold = get_span_threshold(seq_len)
                for batch_spans in all_r:
                    batch_span_indices = []
                    for k, span in enumerate(batch_spans):
                        span_indices = []
                        for l, frame_score in enumerate(span):
                            if frame_score.item() > span_threshold[l]:
                                span_indices.append(l+first_n_frame_dynamics+1)
                        batch_span_indices.append(span_indices)
                    all_span_list.append(batch_span_indices)

            if model_type == 'pip' or model_type == 'ablation':
                print("Batch {}/{} testing done with cls loss={}, cls accuracy={}, gen loss={}, jsd loss={}.".format(j+1, len(test_dataloader), temp_test_classification_loss[-1] / retrieved_batch_size, test_acc, temp_test_image_loss[-1] / retrieved_batch_size, temp_test_jsd_loss[-1] / retrieved_batch_size))
            elif model_type == 'ablation':
                print("Batch {}/{} testing done with cls loss={}, cls accuracy={}, gen loss={}.".format(j+1, len(test_dataloader), temp_test_classification_loss[-1] / retrieved_batch_size, test_acc, temp_test_image_loss[-1] / retrieved_batch_size))
            elif model_type == 'baseline':
                print("Batch {}/{} testing done with cls loss={}, cls accuracy={}.".format(j+1, len(test_dataloader), temp_test_classification_loss[-1] / retrieved_batch_size, test_acc))

        if model_type == 'pip':
            print("OVERALL test cls loss={}, cls accuracy={}, gen loss={}, jsd loss={}.\n".format(sum(temp_test_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_test_image_loss) / total_cnt, sum(temp_test_jsd_loss) / total_cnt))
        elif model_type == 'ablation':
            print("OVERALL test cls loss={}, cls accuracy={}, gen loss={}.\n".format(sum(temp_test_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_test_image_loss) / total_cnt))
        elif model_type == 'baseline':
            print("OVERALL test cls loss={}, cls accuracy={}.\n".format(sum(temp_test_classification_loss) / total_cnt, total_num_correct / total_cnt))
        stats['test']['cls_loss'].append(sum(temp_test_classification_loss) / total_cnt)
        stats['test']['cls_acc'].append(total_num_correct / total_cnt)
        if model_type == 'pip' or model_type == 'ablation':
            stats['test']['gen_loss'].append(sum(temp_test_image_loss) / total_cnt)

        # save frames and spans
        if model_type == 'pip' or model_type == 'ablation':
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

            if model_type == 'pip':
                span_threshold = get_span_threshold(seq_len)
                with open(os.path.join(experiment_save_path, 'spans.txt'), 'w') as f:
                    for k, span in enumerate(all_r[0]):
                        span_indices = []
                        for l, frame_score in enumerate(span):
                            if frame_score.item() > span_threshold[l]:
                                span_indices.append(l+first_n_frame_dynamics+1)
                        f.write('Span {}: '.format(k) + str(span_indices) + '\n')

                print("Saved new test frames and spans.\n")
            elif model_type == 'ablation':
                print("Saved new test frames.\n")

            if save_spans and model_type == 'pip':
                with open(os.path.join(experiment_save_path, 'all_spans.json'), 'w') as f:
                    json.dump(all_span_list, f)
                    f.close()
                print("Saved all selected spans.")

        with open(os.path.join(experiment_save_path, 'log.txt'), 'w') as f:
            f.write('{}\n'.format(cfg))
            f.write('{}\n'.format(stats))
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
    mask_path = cfg['mask_path']
    train_label_path = cfg['train_label_path']
    val_label_path = cfg['val_label_path']
    test_label_path = cfg['test_label_path']
    save_path = cfg['save_path']
    save_spans = cfg['save_spans']
    load_model_path = cfg['load_model_path']
    experiment_type = cfg['experiment_type']
    device = cfg['device']
    model_type = cfg['model_type']
    num_epoch = cfg['num_epoch']
    batch_size = cfg['batch_size']
    teacher_forcing_prob = cfg['teacher_forcing_prob']
    first_n_frame_dynamics = cfg['first_n_frame_dynamics']
    frame_interval = cfg['frame_interval']
    learning_rate = cfg['learning_rate']
    max_seq_len = cfg['max_seq_len']
    span_num = cfg['span_num']
    seed = cfg['seed']

    # check configs
    if task_type != 'contact' and task_type != 'contain' and task_type != 'stability' and task_type != 'combined':
        assert False, "Is your task_type contact, contain, stability or combined?"
    if experiment_type != 'train' and experiment_type != 'test':
        assert False, "Is your experiment_type train or test?"
    if model_type != 'pip' and model_type != 'ablation' and model_type != 'phydnet' and model_type != 'baseline':
        assert False, "Is your model_type pip, ablation, phydnet or baseline?"
    assert num_epoch > 0 and type(num_epoch) == int
    assert batch_size > 0 and type(batch_size) == int
    assert teacher_forcing_prob >= 0 and teacher_forcing_prob <= 1
    assert first_n_frame_dynamics >= 0 and type(first_n_frame_dynamics) == int
    assert frame_interval > 0 and type(frame_interval) == int
    assert learning_rate > 0

    # set seed values
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if experiment_type == 'train':
        if model_type == 'phydnet':
            phydnet_train(cfg, task_type, frame_path, mask_path, train_label_path, val_label_path, test_label_path, save_path, load_model_path, num_epoch, batch_size, first_n_frame_dynamics, frame_interval, save_spans, max_seq_len, device)
        else:
            train(cfg, task_type, frame_path, mask_path, train_label_path, val_label_path, test_label_path, save_path, load_model_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, learning_rate, save_spans, max_seq_len, span_num, device, model_type)
    
    elif experiment_type == 'test':
        if model_type == 'phydnet':
            phydnet_test(cfg, task_type, frame_path, mask_path, train_label_path, val_label_path, test_label_path, save_path, load_model_path, num_epoch, batch_size, first_n_frame_dynamics, frame_interval, save_spans, max_seq_len, device)
        else:
            test(cfg, task_type, frame_path, mask_path, train_label_path, val_label_path, test_label_path, save_path, load_model_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, learning_rate, save_spans, max_seq_len, span_num, device, model_type)