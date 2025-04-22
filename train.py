import os
import ast
import copy
import json
import time
import torch
import random
import logging
import argparse
import traceback

import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from dataset import CRCDataset
from models.crc_model import CRCModel
from utils.losses import DIoULoss
from utils.utils import set_seed, evaluate_model, evaluate_boxes, plot_slices, plot_history


set_seed(seed=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, stu_model, train_loader, val_loader, criterion, criterion_loc):

    # Training record
    best_val = None
    train_epoch_loss, train_epoch_acc, train_epoch_auc, train_epoch_precision, train_epoch_sensitivity, train_epoch_specificity, train_epoch_f1score = [], [], [], [], [], [], []
    train_epoch_loc_precision, train_epoch_loc_sensitivity, train_epoch_iou = [], [], []
    val_epoch_loss, val_epoch_acc, val_epoch_auc, val_epoch_precision, val_epoch_sensitivity, val_epoch_specificity, val_epoch_f1score = [], [], [], [], [], [], []
    val_epoch_loc_precision, val_epoch_loc_sensitivity, val_epoch_iou = [], [], []

    # Length of data loader
    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)

    # Optimizer and learning rate scheduler
    if args.distill:
        optimizer = torch.optim.Adam(stu_model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
        # Knowledge distillation parameters
        alpha = 0.5
        T = 2.0
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    # Localization loss
    beta = 0.5
    gamma = 0.5

    mse_loss = nn.MSELoss(reduction='mean')

    for epoch in range(args.epochs):

        if args.distill:
            model.eval()  # Teacher model
            stu_model.train()  # Student model
        else:
            model.train()
        running_loss = 0
        y_true, y_scores, y_pred, y_true_box, y_pred_box = [], [], [], [], []
        for i, data in enumerate(train_loader, 0):
            # Images, 3D scans, boxes, labels [inputs, volumes, boxes, labels]
            inputs, volumes, boxes, labels = data
            inputs = inputs.to(device)
            volumes = volumes.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            loss_loc = torch.tensor(0)
            if args.distill:
                # Teacher model predictions
                with torch.no_grad():
                    if args.align:
                        if args.loc:
                            teacher_outputs, _, _, _, _ = model(inputs, volumes)
                        else:
                            teacher_outputs, _, _, _ = model(inputs, volumes)
                    else:
                        if args.loc:
                            teacher_outputs, _ = model(inputs, volumes)
                        else:
                            teacher_outputs = model(inputs, volumes)
                # Student model predictions
                if args.align:
                    if args.loc:
                        outputs, outputs_loc, outputs_N, outputs_A, outputs_P = stu_model(inputs, volumes)
                    else:
                        outputs, outputs_N, outputs_A, outputs_P = stu_model(inputs, volumes)
                else:
                    if args.loc:
                        outputs, outputs_loc = stu_model(inputs, volumes)
                    else:
                        outputs = stu_model(inputs, volumes)

                soft_targets = F.softmax(teacher_outputs / T, dim=-1)
                soft_prob = F.log_softmax(outputs / T, dim=-1)
                # Distillation loss with a large T value
                soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)
                # Student loss with T=1
                loss_ce = criterion(outputs, labels)
                # Distillation loss + student loss
                loss_cls = alpha * soft_targets_loss + (1 - alpha) * loss_ce
            else:
                if args.align:
                    if args.loc:
                        outputs, outputs_loc, outputs_N, outputs_A, outputs_P = model(inputs, volumes)
                    else:
                        outputs, outputs_N, outputs_A, outputs_P = model(inputs, volumes)
                else:
                    if args.loc:
                        outputs, outputs_loc = model(inputs, volumes)
                    else:
                        outputs = model(inputs, volumes)
                loss_cls = criterion(outputs, labels)  # Classification loss

            mask_pos = ~torch.all(boxes == torch.tensor([-1.0, -1.0, -1.0, -1.0]).to(device), dim=1)  # Compute localization loss only for cancer slices
            loss_align = torch.tensor(0)
            if args.loc:
                loss_loc = criterion_loc(outputs_loc[mask_pos].float(), boxes[mask_pos].float())  # Localization loss
                y_true_box.extend(boxes[mask_pos].cpu().detach().numpy())
                y_pred_box.extend(outputs_loc[mask_pos].cpu().detach().numpy())
                if args.align:
                    # Classification loss + localization loss + plain-enhanced modal alignment loss
                    loss_align = mse_loss(outputs_N, outputs_A) + mse_loss(outputs_N, outputs_P)
                    loss = loss_cls + beta * loss_loc + gamma * loss_align
                else:
                    # Classification loss + localization loss
                    loss = loss_cls + beta * loss_loc
            else:
                if args.align:
                    # Classification loss + plain-enhanced modal alignment loss
                    loss_align = mse_loss(outputs_N, outputs_A) + mse_loss(outputs_N, outputs_P)
                    loss = loss_cls + gamma * loss_align
                else:
                    # Classification loss
                    loss = loss_cls

            loss.backward()
            optimizer.step()

            Y_prob = F.softmax(outputs, dim=1)
            Y_hat = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().detach().numpy())
            y_scores.extend(Y_prob[:, 1].cpu().detach().numpy())
            y_pred.extend(Y_hat.cpu().detach().numpy())

            running_loss += loss.item() * inputs.size(0)

            # Print status
            if i % args.log_step == 0:
                try:
                    if args.loc:
                        loc_precision, loc_sensitivity, iou = evaluate_boxes(outputs_loc, Y_hat.cpu().detach().numpy(), boxes, labels.cpu().detach().numpy())
                    else:
                        loc_precision, loc_sensitivity, iou = 0, 0, 0
                    acc, auc, precision, sensitivity, specificity, f1score = evaluate_model(labels.cpu().detach().numpy(),
                                                                                            Y_prob[:, 1].cpu().detach().numpy(),
                                                                                            Y_hat.cpu().detach().numpy())
                except ValueError:
                    acc, auc, precision, sensitivity, specificity, f1score = 0, 0, 0, 0, 0

                if args.loc:
                    logging.info(
                        "Training epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, train_loader_len) +
                        "lr:{:.8f} ".format(optimizer.param_groups[0]['lr']) +
                        "acc:{:.3f} ".format(acc) +
                        "auc:{:.3f} ".format(auc) +
                        "precision:{:.3f} ".format(precision) +
                        "sensitivity:{:.3f} ".format(sensitivity) +
                        "specificity:{:.3f} ".format(specificity) +
                        "f1_score:{:.3f} ".format(f1score) +
                        "loc_precision:{:.3f} ".format(loc_precision) +
                        "loc_sensitivity:{:.3f} ".format(loc_sensitivity) +
                        "iou:{:.3f} ".format(iou) +
                        "loss_cls:{:.3f} ".format(loss_cls.item()) +
                        "loss_loc:{:.3f} ".format(loss_loc.item()) +
                        "loss_align:{:.3f} ".format(loss_align.item()) +
                        "loss:{:.3f}".format(loss.item())
                    )
                else:
                    logging.info(
                        "Training epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, train_loader_len) +
                        "lr:{:.8f} ".format(optimizer.param_groups[0]['lr']) +
                        "acc:{:.3f} ".format(acc) +
                        "auc:{:.3f} ".format(auc) +
                        "precision:{:.3f} ".format(precision) +
                        "sensitivity:{:.3f} ".format(sensitivity) +
                        "specificity:{:.3f} ".format(specificity) +
                        "f1_score:{:.3f} ".format(f1score) +
                        "loss_cls:{:.3f} ".format(loss_cls.item()) +
                        "loss:{:.3f}".format(loss.item())
                    )

        scheduler.step()

        # Validation
        if args.distill:
            stu_model.eval()
        else:
            model.eval()
        val_loss = 0
        val_y_true, val_y_scores, val_y_pred, val_y_true_box, val_y_pred_box = [], [], [], [], []
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                # Images, 3D scans, boxes, labels [inputs, volumes, boxes, labels]
                inputs, volumes, boxes, labels = data
                inputs = inputs.to(device)
                volumes = volumes.to(device)
                boxes = boxes.to(device)
                labels = labels.to(device)

                if args.distill:
                    if args.align:
                        if args.loc:
                            outputs, outputs_loc, outputs_N, outputs_A, outputs_P = stu_model(inputs, volumes)
                        else:
                            outputs, outputs_N, outputs_A, outputs_P = stu_model(inputs, volumes)
                    else:
                        if args.loc:
                            outputs, outputs_loc = stu_model(inputs, volumes)
                        else:
                            outputs = stu_model(inputs, volumes)
                else:
                    if args.align:
                        if args.loc:
                            outputs, outputs_loc, outputs_N, outputs_A, outputs_P = model(inputs, volumes)
                        else:
                            outputs, outputs_N, outputs_A, outputs_P = model(inputs, volumes)
                    else:
                        if args.loc:
                            outputs, outputs_loc = model(inputs, volumes)
                        else:
                            outputs = model(inputs, volumes)
                Y_prob = F.softmax(outputs, dim=1)
                Y_hat = torch.argmax(outputs, dim=1)

                # Classification loss
                loss_cls = criterion(outputs, labels)
                # Localization loss
                loss_loc = torch.tensor(0)
                # Alignment loss
                loss_align = torch.tensor(0)

                mask_pos = ~torch.all(boxes == torch.tensor([-1.0, -1.0, -1.0, -1.0]).to(device), dim=1)  # Cancer slices
                if args.loc:
                    # Localization loss
                    loss_loc = criterion_loc(outputs_loc[mask_pos].float(), boxes[mask_pos].float())
                    if args.align:
                        if torch.isnan(loss_loc):  # Handle validation sets with all-negative inputs
                            loss = loss_cls + gamma * loss_align
                        else:
                            loss = loss_cls + beta * loss_loc + gamma * loss_align
                    else:
                        if torch.isnan(loss_loc):  # Handle validation sets with all-negative inputs
                            loss = loss_cls
                        else:
                            loss = loss_cls + beta * loss_loc

                    val_y_true_box.extend(boxes[mask_pos].cpu().detach().numpy())
                    val_y_pred_box.extend(outputs_loc[mask_pos].cpu().detach().numpy())
                else:
                    if args.align:
                        loss = loss_cls + gamma * loss_align
                    else:
                        loss = loss_cls

                val_y_true.extend(labels.cpu().detach().numpy())
                val_y_scores.extend(Y_prob[:, 1].cpu().detach().numpy())
                val_y_pred.extend(Y_hat.cpu().detach().numpy())

                val_loss += loss.item() * inputs.size(0)

                if i % args.log_step == 0:
                    logging.info(
                        "Validating epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, val_loader_len) +
                        "loss_cls:{:.3f} ".format(loss_cls.item()) +
                        "loss_loc:{:.3f} ".format(loss_loc.item()) +
                        "loss_align:{:.3f} ".format(loss_align.item()) +
                        "loss:{:.3f}".format(loss.item())
                    )

        # Summary for the epoch
        acc, auc, precision, sensitivity, specificity, f1score = evaluate_model(y_true, y_scores, y_pred)
        val_acc, val_auc, val_precision, val_sensitivity, val_specificity, val_f1score = evaluate_model(val_y_true, val_y_scores, val_y_pred)

        if args.loc:
            loc_precision, loc_sensitivity, iou = evaluate_boxes(y_pred_box, y_pred, y_true_box, y_true)
            val_loc_precision, val_loc_sensitivity, val_iou = evaluate_boxes(val_y_pred_box, val_y_pred, val_y_true_box,  val_y_true)
        else:
            loc_precision, loc_sensitivity, iou = 0, 0, 0
            val_loc_precision, val_loc_sensitivity, val_iou = 0, 0, 0

        logging.info(
            "[Epoch:{:<5}/{:<5}] ".format(epoch+1, args.epochs) +
            "acc:{:.3f} auc:{:.3f} precision:{:.3f} sensitivity:{:.3f} specificity:{:.3f} f1_score:{:.3f} loss:{:.3f} ".format(
                acc, auc, precision, sensitivity, specificity, f1score, running_loss / len(train_loader.dataset))
        )
        logging.info(
            "[Epoch:{:<5}/{:<5}] ".format(epoch+1, args.epochs) +
            "val_acc:{:.3f} val_auc:{:.3f} val_precision:{:.3f} val_sensitivity:{:.3f} val_specificity:{:.3f} val_f1_score:{:.3f} val_loss:{:.3f} ".format(
                val_acc, val_auc, val_precision, val_sensitivity, val_specificity, val_f1score, val_loss / len(val_loader.dataset))
        )
        if args.loc:
            logging.info(
                "[Epoch:{:<5}/{:<5}] ".format(epoch+1, args.epochs) +
                "loc_precision:{:.3f} loc_sensitivity:{:.3f} iou:{:.3f} ".format(loc_precision, loc_sensitivity, iou) +
                "val_loc_precision:{:.3f} val_loc_sensitivity:{:.3f} val_iou:{:.3f}".format(val_loc_precision, val_loc_sensitivity, val_iou)
            )

        # Train metrics
        train_epoch_acc.append(acc)
        train_epoch_auc.append(auc)
        train_epoch_precision.append(precision)
        train_epoch_sensitivity.append(sensitivity)
        train_epoch_specificity.append(specificity)
        train_epoch_f1score.append(f1score)
        train_epoch_loc_precision.append(loc_precision)
        train_epoch_loc_sensitivity.append(loc_sensitivity)
        train_epoch_iou.append(iou)
        train_epoch_loss.append(running_loss / len(train_loader.dataset))
        # Validation metrics
        val_epoch_acc.append(val_acc)
        val_epoch_auc.append(val_auc)
        val_epoch_precision.append(val_precision)
        val_epoch_sensitivity.append(val_sensitivity)
        val_epoch_specificity.append(val_specificity)
        val_epoch_f1score.append(val_f1score)
        val_epoch_loc_precision.append(val_loc_precision)
        val_epoch_loc_sensitivity.append(val_loc_sensitivity)
        val_epoch_iou.append(val_iou)
        val_epoch_loss.append(val_loss / len(val_loader.dataset))
        plot_history(train_epoch_acc, train_epoch_auc, train_epoch_precision, train_epoch_sensitivity, train_epoch_specificity, train_epoch_f1score,
                     train_epoch_loc_precision, train_epoch_loc_sensitivity, train_epoch_iou, train_epoch_loss,
                     val_epoch_acc, val_epoch_auc, val_epoch_precision, val_epoch_sensitivity, val_epoch_specificity, val_epoch_f1score,
                     val_epoch_loc_precision, val_epoch_loc_sensitivity, val_epoch_iou, val_epoch_loss, history_save_path)

        # Save checkpoint
        model_save = copy.deepcopy(stu_model) if args.distill else copy.deepcopy(model)
        if best_val is None or val_acc > best_val:
            best_val = val_acc
            torch.save(model_save.state_dict(), os.path.join(ckpt_path, "best.pth"))
            logging.info("Saved best model.")

    logging.info('STOP TIME:{}'.format(time.asctime(time.localtime(time.time()))))


def main():
    # Initialize the model
    model = CRCModel(model_name=args.model_name, loc=args.loc, align=args.align)

    # Parallel processing
    model = nn.DataParallel(model)
    model.to(device)

    if args.pretrain_weight_path:
        model.load_state_dict(torch.load(args.pretrain_weight_path), strict=False)

    logging.info('START TIME:{}'.format(time.asctime(time.localtime(time.time()))))
    logging.info(vars(args))

    # Late-onset colorectal cancer teacher network NAP → Early-onset colorectal cancer N
    crc_train_path, crc_val_path = [], []
    if args.mode == 'N':  # Single-modality student model testing or teacher model distillation
        for line in open('data/EOLO_new/crc_eo_train.txt'):
            crc_train_path.append(ast.literal_eval(line.strip()))
    else:  # Teacher model
        for line in open('data/EOLO_new/crc_lo_train.txt'):
            crc_train_path.append(ast.literal_eval(line.strip()))
    for line in open('data/EOLO_new/crc_eo_val.txt'):
        crc_val_path.append(ast.literal_eval(line.strip()))

    # Use teacher model NAP to distill student model N
    if args.distill:
        model.load_state_dict(torch.load(args.pretrain_weight_path))
        stu_model = CRCModel(model_name=args.model_name, loc=args.loc)
        stu_model = nn.DataParallel(stu_model)
        stu_model.to(device)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.GaussNoise(var_limit=0.01, p=0.5),
        A.OneOf([
            A.GridDistortion(p=0.5),
            A.ElasticTransform(p=0.5),
        ], p=0.75)
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))

    train_dataset = CRCDataset(crc_train_path, mode=args.mode, transform=train_transform, use_volume=args.use_volume, align=args.align, type='train')
    if args.exp_name.find('teacher') != -1 or args.exp_name.find('student') != -1:
        val_dataset = CRCDataset(crc_val_path, mode='N', use_volume=args.use_volume, align=args.align, type='val')  # teacher or student
    else:
        val_dataset = CRCDataset(crc_val_path, mode=args.mode, use_volume=args.use_volume, align=args.align, type='val')

    # Dataset sampler
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=24, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24, pin_memory=True)

    logging.info("Train slices: {}, Val slices: {}".format(len(train_loader.dataset), len(val_loader.dataset)))

    if args.plt:
        idx = 0
        image, volume, label = train_dataset.__getitem__(idx)
        print("CT image:{}, volume:{}, label:{}".format(image.shape, volume.shape, label))
        image = np.squeeze(image)  # Image
        volume = np.squeeze(volume)
        # Plot a single CT cross-section
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        plt.savefig('figure/CT.png', bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close()
        # Plot 3D scan
        plot_slices(num_rows=1, num_columns=7, width=args.img_size, height=args.img_size, scan_mode=args.mode, data=volume)

    # Binary cross-entropy loss function
    criterion = nn.CrossEntropyLoss()
    # Localization loss function
    criterion_loc = DIoULoss()

    # Train the model
    if args.distill:
        logging.info('Distill Model')
        train(model, stu_model, train_loader, val_loader, criterion, criterion_loc)
    else:
        logging.info('Train Model')
        train(model, None, train_loader, val_loader, criterion, criterion_loc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Colorectal Cancer Classification")
    parser.add_argument("--epochs", type=int, default=20,
                        help="training epochs")
    parser.add_argument("--img_size", type=int, default=224,
                        help="image size")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="initial learning rate")
    parser.add_argument("--log_step", type=int, default=300,
                        help="log accuracy each log_step batchs")
    parser.add_argument("--model_name", type=str, default="convnextv2_large.fcmae_ft_in22k_in1k",
                        help="model name")
    parser.add_argument("--mode", type=str, default='NAP',
                        help="mode to train")
    parser.add_argument("--use_volume", action="store_true", default=False,
                        help="use volume")
    parser.add_argument("--loc", action="store_true", default=False,
                        help="lesion localization")
    parser.add_argument("--distill", action="store_true", default=False,
                        help="knowledge distillation")
    parser.add_argument("--align", action="store_true", default=False,
                        help="feature alignment")
    parser.add_argument("--plt", action="store_true", default=False,
                        help="plot CT image")
    parser.add_argument("--pretrain_weight_path", type=str, default="",
                        help="pretrain weight path")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="experiment name")

    args = parser.parse_args()

    ckpt_path = os.path.join('run/ckpt', args.exp_name)
    logs_path = os.path.join('run/logs', args.exp_name, "{}-{}.log".format(datetime.now().strftime("%Y-%m-%d_%H.%M.%S"), args.exp_name))
    history_save_path = os.path.join('run/history', args.exp_name, "{}-{}.png".format(datetime.now().strftime("%Y-%m-%d_%H.%M.%S"), args.exp_name))

    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(os.path.join('run/logs', args.exp_name), exist_ok=True)
    os.makedirs(os.path.join('run/history', args.exp_name), exist_ok=True)

    # Define log format
    log_format = "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create file handler and stream handler
    file_handler = logging.FileHandler(logs_path, mode='a')
    stream_handler = logging.StreamHandler()

    # Create formatter and add it to handlers
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Get the root logger and set level
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        # To avoid duplicate logging, clear existing handlers and re-add
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    try:
        main()
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        exit(1)
