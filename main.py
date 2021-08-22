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


def get_classification_accuracy(pred_labels, labels):
    """
    Get accuracy for classification.
    """
    size = pred_labels.shape[0]
    mask = pred_labels >= 0
    num_correct = torch.sum(mask == labels).item()
    acc = num_correct / size

    return acc, num_correct


def train(cfg, task_type, frame_path, mask_path, train_label_path, val_label_path, test_label_path, save_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, learning_rate, save_frames_every, max_seq_len, span_num, span_threshold, jsd_theta):
    # get experiment ID
    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +  ' ' + task_type
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    experiment_save_path = os.path.join(save_path, experiment_id)
    os.makedirs(experiment_save_path, exist_ok=True)
    
    if task_type == 'combined':
        train_combine_idx = {'contact': [1,200], 'contain': [201,400], 'stability': [401,600]}
        train_dataset = Data(frame_path, mask_path, train_label_path, frame_interval, first_n_frame_dynamics, task_type, combined_scene_tasks=train_combine_idx)
    else:
        train_dataset = Data(frame_path, mask_path, train_label_path, frame_interval, first_n_frame_dynamics, task_type)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    if task_type == 'combined':
        val_combine_idx = {'contact': [601,666], 'contain': [667,733], 'stability': [734,800]}
        val_dataset = Data(frame_path, mask_path, val_label_path, frame_interval, first_n_frame_dynamics, task_type, combined_scene_tasks=val_combine_idx)
    else:
        val_dataset = Data(frame_path, mask_path, val_label_path, frame_interval, first_n_frame_dynamics, task_type)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # NOTE
    save_img_dir = os.path.join(experiment_save_path, 'generations')

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = Model(device, span_num, jsd_theta, nc=3, nf=16).to(device)
    # turn off gradients for other tasks
    tasks = ['contact', 'contain', 'stability']
    for i in tasks:
        if i != task_type:
            for n, p in model.named_parameters():
                if i in n:
                    p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bce_logits_loss = nn.BCEWithLogitsLoss().to(device)
    bce_loss = nn.BCELoss().to(device)
    frame_loss = PSNR().to(device)

    stats = {'train': {'cls_loss': [], 'cls_acc': [], 'gen_loss': []}, 'val': {'cls_loss': [], 'cls_acc': [], 'gen_loss': []}}
    
    min_val_classification_loss = math.inf
    max_val_image_loss = 0
    min_val_classification_epoch = None
    max_val_image_epoch = None

    for i in range(num_epoch):
        # training
        print('Training for epoch {}/{}...'.format(i+1, num_epoch))
        temp_train_classification_loss = []
        temp_train_image_loss = []
        temp_train_jsd_loss = []
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
            loss = cls_loss + torch.mean(jsd_loss)

            temp_train_classification_loss.append(cls_loss.data.item() * retrieved_batch_size)
            temp_train_image_loss.append(0)
            temp_train_jsd_loss.append(torch.mean(jsd_loss).data.item() * retrieved_batch_size)

            # save generated images for testing
            if save_frames_every is not None and j % save_frames_every == 0:
                epoch_train_save_img_dir = os.path.join(os.path.join(save_img_dir, str(i)), 'train')
                pred_save_img_dir = os.path.join(epoch_train_save_img_dir, 'pred')
                real_save_img_dir = os.path.join(epoch_train_save_img_dir, 'real')
                os.makedirs(pred_save_img_dir, exist_ok=True)
                os.makedirs(real_save_img_dir, exist_ok=True)
                for k in range(first_n_frame_dynamics):
                    save_image(frames[k][0], os.path.join(pred_save_img_dir, '{}.png'.format(k)))
                    save_image(frames[k][0], os.path.join(real_save_img_dir, '{}.png'.format(k)))
                    save_image(masks[k][0], os.path.join(pred_save_img_dir, '{}_mask.png'.format(k)))
                    save_image(masks[k][0], os.path.join(real_save_img_dir, '{}_mask.png'.format(k)))

            for k, pred_images in enumerate(pred_images_seq[:-1]):
                # save generated images for testing
                if save_frames_every is not None and j % save_frames_every == 0:
                    save_image(pred_images[0], os.path.join(pred_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics)))
                    save_image(frames[k+first_n_frame_dynamics][0], os.path.join(real_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics)))

                frames_k = frames[k+first_n_frame_dynamics].to(device)
                pred_images = torch.clamp(pred_images, 0, 1)
                img_loss = frame_loss(pred_images, frames_k)
                loss += -img_loss
                temp_train_image_loss[-1] += img_loss.data.item() * retrieved_batch_size
                seq_len = len(pred_images_seq[:-1])
            
            # save selected spans
            with open(os.path.join(epoch_train_save_img_dir, 'spans.txt'), 'w') as f:
                for k, span in enumerate(all_r[0]):
                    span_indices = []
                    for l, frame_score in enumerate(span):
                        if frame_score.item() > span_threshold:
                            span_indices.append(l+first_n_frame_dynamics)
                    f.write('Span {}: '.format(k) + str(span_indices) + '\n')
                f.write('\n')
                for k, span in enumerate(all_r[0]):
                    f.write('Span {}: '.format(k) + str(span) + '\n')
                f.close()

            if save_frames_every is not None and j % save_frames_every == 0:
                if teacher_forcing_batch[0]:
                    print("Saved new train sequences WITH teacher forcing.")
                else:
                    print("Saved new train sequences WITHOUT teacher forcing.")

            temp_train_image_loss[-1] /= seq_len

            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch {}/{} batch {}/{} training done with cls loss={}, cls accuracy={}, gen loss={}, jsd loss={}.".format(i+1, num_epoch, j+1, len(train_dataloader), temp_train_classification_loss[-1] / retrieved_batch_size, train_acc, temp_train_image_loss[-1] / retrieved_batch_size, temp_train_jsd_loss[-1] / retrieved_batch_size))
            # print("Epoch {}/{} batch {}/{} training done with gen loss={}.".format(i+1, num_epoch, j+1, len(train_dataloader), temp_train_image_loss[-1] / retrieved_batch_size))

        print("Epoch {}/{} OVERALL train cls loss={}, cls accuracy={}, gen loss={}, jsd loss={}.\n".format(i+1, num_epoch, sum(temp_train_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_train_image_loss) / total_cnt, sum(temp_train_jsd_loss), total_cnt))
        stats['train']['cls_loss'].append(sum(temp_train_classification_loss) / total_cnt)
        stats['train']['cls_acc'].append(total_num_correct / total_cnt)
        stats['train']['gen_loss'].append(sum(temp_train_image_loss) / total_cnt)


        # validation
        print('Validation for epoch {}/{}...'.format(i+1, num_epoch))
        temp_val_classification_loss = []
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
                temp_val_image_loss.append(0)

                # save generated images for testing
                if save_frames_every is not None and i % save_frames_every == 0:
                    epoch_val_save_img_dir = os.path.join(os.path.join(save_img_dir, str(i)), 'val')
                    pred_save_img_dir = os.path.join(epoch_val_save_img_dir, 'pred')
                    real_save_img_dir = os.path.join(epoch_val_save_img_dir, 'real')
                    os.makedirs(pred_save_img_dir, exist_ok=True)
                    os.makedirs(real_save_img_dir, exist_ok=True)
                    for k in range(first_n_frame_dynamics):
                        save_image(frames[k][0], os.path.join(pred_save_img_dir, '{}.png'.format(k)))
                        save_image(frames[k][0], os.path.join(real_save_img_dir, '{}.png'.format(k)))
                        save_image(masks[k][0], os.path.join(pred_save_img_dir, '{}_mask.png'.format(k)))
                        save_image(masks[k][0], os.path.join(real_save_img_dir, '{}_mask.png'.format(k)))

                for k, pred_images in enumerate(pred_images_seq[:-1]):
                    # save generated images for testing
                    if save_frames_every is not None and i % save_frames_every == 0:
                        save_image(pred_images[0], os.path.join(pred_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics)))
                        save_image(frames[k+first_n_frame_dynamics][0], os.path.join(real_save_img_dir, '{}.png'.format(k+first_n_frame_dynamics)))
                    
                    frames_k = frames[k+first_n_frame_dynamics].to(device)
                    pred_images = torch.clamp(pred_images, 0, 1)
                    img_loss = frame_loss(pred_images, frames_k)
                    temp_val_image_loss[-1] += img_loss.data.item() * retrieved_batch_size
                    seq_len = len(pred_images_seq[:-1])

                # save selected spans
                with open(os.path.join(epoch_val_save_img_dir, 'spans.txt'), 'w') as f:
                    for k, span in enumerate(all_r[0]):
                        span_indices = []
                        for l, frame_score in enumerate(span):
                            if frame_score.item() > span_threshold:
                                span_indices.append(l+first_n_frame_dynamics)
                        f.write('Span {}: '.format(k) + str(span_indices) + '\n')
                    f.write('\n')
                    for k, span in enumerate(all_r[0]):
                        f.write('Span {}: '.format(k) + str(span) + '\n')
                    f.close()

                temp_val_image_loss[-1] /= seq_len

                print("Saved new validation sequences.")
                
                print("Epoch {}/{} batch {}/{} validation done with cls loss={}, cls accuracy={}, gen loss={}.".format(i+1, num_epoch, j+1, len(val_dataloader), temp_val_classification_loss[-1] / retrieved_batch_size, val_acc, temp_val_image_loss[-1] / retrieved_batch_size))
                # print("Epoch {}/{} batch {}/{} validation done with gen loss={}.".format(i+1, num_epoch, j+1, len(val_dataloader), temp_val_image_loss[-1] / retrieved_batch_size))

        print("Epoch {}/{} OVERALL validation cls loss={}, cls accuracy={}, gen loss={}.\n".format(i+1, num_epoch, sum(temp_val_classification_loss) / total_cnt, total_num_correct / total_cnt, sum(temp_val_image_loss) / total_cnt))
        stats['val']['cls_loss'].append(sum(temp_val_classification_loss) / total_cnt)
        stats['val']['cls_acc'].append(total_num_correct / total_cnt)
        stats['val']['gen_loss'].append(sum(temp_val_image_loss) / total_cnt)

        # check for best stat/model using validation stats
        if stats['val']['cls_loss'][-1] < min_val_classification_loss:
            min_val_classification_loss = stats['val']['cls_loss'][-1]
            min_val_classification_epoch = i
            torch.save(model, os.path.join(experiment_save_path, 'model'))
        if stats['val']['gen_loss'][-1] > max_val_image_loss:
            max_val_image_loss = stats['val']['gen_loss'][-1]
            max_val_image_epoch = i
            # torch.save(model, os.path.join(experiment_save_path, 'model'))

        with open(os.path.join(experiment_save_path, 'log.txt'), 'w') as f:
            f.write('{}\n'.format(cfg))
            f.write('{}\n'.format(stats))
            f.write('Min cls loss: epoch {}, {}\n'.format(min_val_classification_epoch, min_val_classification_loss))
            f.write('Min val gen loss: epoch {}, {}\n'.format(max_val_image_epoch, max_val_image_loss))
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
    save_frames_every = cfg['save_frames_every']
    load_model_path = cfg['load_model_path']
    experiment_type = cfg['experiment_type']
    num_epoch = cfg['num_epoch']
    batch_size = cfg['batch_size']
    teacher_forcing_prob = cfg['teacher_forcing_prob']
    first_n_frame_dynamics = cfg['first_n_frame_dynamics']
    frame_interval = cfg['frame_interval']
    learning_rate = cfg['learning_rate']
    max_seq_len = cfg['max_seq_len']
    span_num = cfg['span_num']
    # TODO: check on span threshold
    span_threshold = cfg['span_threshold']
    jsd_theta = cfg['jsd_theta']

    # check configs
    if task_type != 'contact' and task_type != 'contain' and task_type != 'stability' and task_type != 'combined':
        assert False, "Is your task_type contact, contain, stability or combined??"
    if experiment_type != 'train' and experiment_type != 'finetune' and experiment_type != 'test':
        assert False, "Is your experiment_type train, finetune or test?"
    assert num_epoch > 0 and type(num_epoch) == int
    assert batch_size > 0 and type(batch_size) == int
    assert teacher_forcing_prob >= 0 and teacher_forcing_prob <= 1
    assert first_n_frame_dynamics >= 0 and type(first_n_frame_dynamics) == int
    assert frame_interval > 0 and type(frame_interval) == int
    assert learning_rate > 0
    assert jsd_theta >= 0 and jsd_theta <= 0.5


    if experiment_type == 'train':
        train(cfg, task_type, frame_path, mask_path, train_label_path, val_label_path, test_label_path, save_path, num_epoch, batch_size, teacher_forcing_prob, first_n_frame_dynamics, frame_interval, learning_rate, save_frames_every, max_seq_len, span_num, span_threshold, jsd_theta)
    elif experiment_type == 'finetune':
        pass