import ast
import os
import copy
import json
import time
import torch
import logging
import argparse
import traceback

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import matplotlib.pyplot as plt

from datetime import datetime
from torchsampler import ImbalancedDatasetSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import LNMDataset, LNMDualDataset, LNMDemoDataset, LNMDualDemoDataset, LNM3DDataset
from utils.utils import set_seed, evaluate, evaluate_iou, plot_slices
from models.lnm_model import LNMModel, LNMDualModel, LNM3DModel
from models.config import LABEL_LIST

set_seed(seed=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_lnm_train = 'data/LNM/dataset/lnm_train.txt'
data_lnm_val = 'data/LNM/dataset/lnm_val.txt'
data_lnm_test = 'data/LNM/dataset/lnm_test_filter.txt'


def train():
    if args.threeD:
        model = LNM3DModel()
    elif args.dual:
        model = LNMDualModel(model_name=args.model_name, num_classes=len(LABEL_LIST))
    else:
        model = LNMModel(model_name=args.model_name, num_classes=len(LABEL_LIST), loc=args.loc)

    model = nn.DataParallel(model)
    model.to(device)

    if args.pretrain_weight_path:
        model.load_state_dict(torch.load(args.pretrain_weight_path), strict=False)

    logging.info('START TIME:{}'.format(time.asctime(time.localtime(time.time()))))
    logging.info(vars(args))

    if args.loc:
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.GaussNoise(var_limit=0.01, p=0.5),
            A.OneOf([
                A.GridDistortion(p=0.5),
                A.ElasticTransform(p=0.5),
            ], p=0.75)
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
    else:
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.GaussNoise(var_limit=0.01, p=0.5),
            A.OneOf([
                A.GridDistortion(p=0.5),
                A.ElasticTransform(p=0.5),
            ], p=0.75)
        ])

    lnm_train_path, lnm_val_path = [], []
    for line in open(data_lnm_train):
        lnm_train_path.append(ast.literal_eval(line.strip()))
    for line in open(data_lnm_val):
        lnm_val_path.append(ast.literal_eval(line.strip()))

    if args.threeD:
        train_dataset = LNM3DDataset(lnm_train_path, mode=args.mode, crop=args.crop, type='train')
        val_dataset = LNM3DDataset(lnm_val_path, mode=args.mode, crop=args.crop, type='val')
    elif args.dual:
        train_dataset = LNMDualDataset(lnm_train_path, mode=args.mode, transform=train_transform, type='train')
        val_dataset = LNMDualDataset(lnm_val_path, mode=args.mode, type='val')
    else:
        train_dataset = LNMDataset(lnm_train_path, mode=args.mode, transform=train_transform, crop=args.crop, loc=args.loc, type='train')
        val_dataset = LNMDataset(lnm_val_path, mode=args.mode, crop=args.crop, loc=args.loc, type='val')

    if args.plt:
        idx = 0
        image, label = train_dataset.__getitem__(idx)
        print("CT image:{}, label:{}".format(image.shape, label))
        image = np.squeeze(image)
        if args.threeD:
            plot_slices(2, 4, 224, 224, 'A', image, 'explore_crop')
        else:
            plt.axis('off')
            plt.imshow(image, cmap='gray')
            plt.savefig('figure/CT.png', bbox_inches='tight', pad_inches=0, dpi=500)
            plt.close()

    # imbalanced dataset sampler
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(train_dataset), num_workers=24, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24, pin_memory=True)

    if not args.threeD:
        logging.info("Train slices: {}, Val slices: {}".format(len(train_loader.dataset), len(val_loader.dataset)))

    # cls
    criterion = nn.CrossEntropyLoss()
    # loc
    criterion_loc = nn.SmoothL1Loss()

    # train record
    best_val = None
    train_epoch_loss, train_epoch_acc = [], []
    val_epoch_loss, val_epoch_acc = [], []

    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    scheduler = CosineAnnealingLR(optimizer, args.epochs)

    logging.info('Train Model')

    for epoch in range(args.epochs):

        model.train()  # Train

        running_loss = 0
        y_true, y_pred = [], []
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            if args.loc:
                loss = criterion_loc(outputs, labels)
                y_true.extend(labels.cpu().detach().numpy())
                y_pred.extend(outputs.cpu().detach().numpy())
            else:
                loss = criterion(outputs, labels)

                Y_hat = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().detach().numpy())
                y_pred.extend(Y_hat.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if i % args.log_step == 0:
                if args.loc:
                    # print('Train', y_pred[-1], y_true[-1])
                    iou, recall = evaluate_iou(y_pred, y_true)
                    logging.info(
                        "Training epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, train_loader_len) +
                        "lr:{:.8f} ".format(optimizer.param_groups[0]['lr']) +
                        "iou:{:.3f} ".format(iou) +
                        "recall:{:.3f} ".format(recall) +
                        "loss:{:.3f}".format(loss.item())
                    )
                else:
                    acc, precision, recall, specificity, f1score = evaluate(y_true, y_pred)
                    logging.info(
                        "Training epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, train_loader_len) +
                        "lr:{:.8f} ".format(optimizer.param_groups[0]['lr']) +
                        "acc:{:.2f} ".format(acc) +
                        "precision:{} ".format(precision) +
                        "recall:{} ".format(recall) +
                        "specificity:{} ".format(specificity) +
                        "f1_score:{} ".format(f1score) +
                        "loss:{:.3f}".format(loss.item())
                    )

        scheduler.step()

        model.eval()  # Val

        val_loss = 0
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                if args.loc:
                    loss = criterion_loc(outputs, labels)
                    val_y_true.extend(labels.cpu().detach().numpy())
                    val_y_pred.extend(outputs.cpu().detach().numpy())
                else:
                    loss = criterion(outputs, labels)

                    Y_hat = torch.argmax(outputs, dim=1)
                    val_y_true.extend(labels.cpu().detach().numpy())
                    val_y_pred.extend(Y_hat.cpu().detach().numpy())

                val_loss += loss.item() * inputs.size(0)

                if i % args.log_step == 0:
                    if args.loc:
                        # print('Val', val_y_pred[-1], val_y_true[-1])
                        iou, recall = evaluate_iou(val_y_pred, val_y_true)
                        logging.info(
                            "Validating epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, val_loader_len) +
                            "iou:{:.3f} ".format(iou) +
                            "recall:{:.3f} ".format(recall) +
                            "loss:{:.3f}".format(loss.item())
                        )
                    else:
                        acc, precision, recall, specificity, f1score = evaluate(val_y_true, val_y_pred)
                        logging.info(
                            "Validating epoch:{}/{} batch:{}/{} ".format(epoch+1, args.epochs, i+1, val_loader_len) +
                            "acc:{:.2f} ".format(acc) +
                            "precision:{} ".format(precision) +
                            "recall:{} ".format(recall) +
                            "specificity:{} ".format(specificity) +
                            "f1_score:{} ".format(f1score) +
                            "loss:{:.3f}".format(loss.item())
                        )

        # epoch summary
        if args.loc:
            acc, recall = evaluate_iou(y_pred, y_true)
            val_acc, val_recall = evaluate_iou(val_y_pred, val_y_true)

            logging.info(
                "[Epoch:{:<5}/{:<5}] ".format(epoch+1, args.epochs) +
                "iou:{:.3f} recall:{:.3f} loss:{:.3f} ".format(acc, recall, running_loss / len(train_loader.dataset))
            )
            logging.info(
                "[Epoch:{:<5}/{:<5}] ".format(epoch+1, args.epochs) +
                "val_iou:{:.3f} val_recall:{:.3f} val_loss:{:.3f} ".format(val_acc, val_recall, val_loss / len(val_loader.dataset))
            )
        else:
            acc, precision, recall, specificity, f1score = evaluate(y_true, y_pred)
            val_acc, val_precision, val_recall, val_specificity, val_f1score = evaluate(val_y_true, val_y_pred)

            logging.info(
                "[Epoch:{:<5}/{:<5}] ".format(epoch+1, args.epochs) +
                "acc:{:.2f} precision{} recall:{} specificity:{} f1_score:{} loss:{:.3f} ".format(
                    acc, precision, recall, specificity, f1score, running_loss / len(train_loader.dataset))
            )
            logging.info(
                "[Epoch:{:<5}/{:<5}] ".format(epoch+1, args.epochs) +
                "val_acc:{:.2f} val_precision:{} val_recall:{} val_specificity:{} val_f1_score:{} val_loss:{:.3f} ".format(
                    val_acc, val_precision, val_recall, val_specificity, val_f1score, val_loss / len(val_loader.dataset))
            )

        # Train
        train_epoch_acc.append(acc)
        train_epoch_loss.append(running_loss / len(train_loader.dataset))
        # Val
        val_epoch_acc.append(val_acc)
        val_epoch_loss.append(val_loss / len(val_loader.dataset))

        _, ax = plt.subplots(1, 2, figsize=(8, 4))
        # ACC
        ax[0].plot(train_epoch_acc)
        ax[0].plot(val_epoch_acc)
        ax[0].set_title("Model {}".format("ACC"))
        ax[0].set_xlabel("epochs")
        ax[0].legend(["train", "val"])
        # Loss
        ax[1].plot(train_epoch_loss)
        ax[1].plot(val_epoch_loss)
        ax[1].set_title("Model {}".format("Loss"))
        ax[1].set_xlabel("epochs")
        ax[1].legend(["train", "val"])
        plt.savefig(history_save_path)
        plt.close()

        # save checkpoint
        model_save = copy.deepcopy(model)
        if best_val is None or val_acc > best_val:
            best_val = val_acc
            torch.save(model_save.state_dict(), os.path.join(ckpt_path, "best.pth"))
            logging.info("Saved best model.")
        torch.save(model_save.state_dict(), os.path.join(ckpt_path, "model_{}.pth".format(epoch + 1)))

    logging.info('STOP TIME:{}'.format(time.asctime(time.localtime(time.time()))))


def test():
    test_path = 'run/lnm_test'

    if args.threeD:
        base_model = LNM3DModel()
    elif args.dual:
        base_model = LNMDualModel(model_name=args.model_name, num_classes=len(LABEL_LIST))
    else:
        base_model = LNMModel(model_name=args.model_name, num_classes=2)
    base_model = nn.DataParallel(base_model)

    base_path = 'run/ckpt/PCls'

    model_path = os.path.join(base_path, args.model)
    assert os.path.exists(model_path)

    state_dict = torch.load(model_path)
    model = base_model
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    lnm_test_path = []

    cls_threshold = 0.5
    count_ones_threshold = 1

    for line in open(args.data_path):
        lnm_test_path.append(ast.literal_eval(line.strip()))

    model_name = base_path.split('/')[-1] + '_' + model_path.split('/')[-1].split('.')[0]
    demo_save_path = os.path.join(test_path, base_path.split('/')[-1])
    os.makedirs(demo_save_path, exist_ok=True)
    print(model_name)

    demo_results_log_path = os.path.join(demo_save_path, 'demo_results.log')

    if args.threeD:
        y_true, y_pred = [], []
        return  # 3D Volume
    else:
        # slice-level
        if args.dual:
            test_dataset = LNMDualDataset(lnm_test_path, mode=args.mode, type='test')
        else:
            test_dataset = LNMDataset(lnm_test_path, mode=args.mode, crop=args.crop, loc=args.loc, type='test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24, pin_memory=True)
        print("Test slices: {}".format(len(test_loader.dataset)))

        y_true, y_pred = [], []
        total = len(test_loader)
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                print("{:<5}/{:<5}".format(i+1, total), end="\r")
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                Y_hat = (F.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy() >= cls_threshold).astype(int)

                y_true.extend(labels.cpu().detach().numpy())
                y_pred.extend(Y_hat)

        acc, precision, recall, specificity, f1score = evaluate(y_true, y_pred)
        print("slice-level acc:{:.2f} precision:{} recall:{} specificity:{} f1_score:{}\n".format(acc, precision, recall, specificity, f1score))

        with open(demo_results_log_path, 'a+') as f:
            f.write('model_name:{}\nslice-level acc:{:.2f} precision:{} recall:{} specificity:{} f1_score:{}\n'.format(
                model_name, acc, precision, recall, specificity, f1score))
        print('-' * 50 + '\n')

        # patient-level
        if args.dual:
            test_dataset_demo = LNMDualDemoDataset(lnm_test_path, mode=args.mode)
        else:
            test_dataset_demo = LNMDemoDataset(lnm_test_path, mode=args.mode, crop=args.crop)

        test_loader_demo = torch.utils.data.DataLoader(test_dataset_demo, batch_size=1, shuffle=False, num_workers=24, pin_memory=True)
        y_true, y_pred = [], []
        total = len(test_loader_demo)
        with torch.no_grad():
            for i, data in enumerate(test_loader_demo, 0):
                print("{:<5}/{:<5}".format(i+1, total), end="\r")
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                Y_hat = (F.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy() >= cls_threshold).astype(int)

                count_ones = np.sum(Y_hat == 1).item()
                if count_ones >= count_ones_threshold:
                    y_pred_label = 1
                else:
                    y_pred_label = 0

                y_pred_label = np.bincount(Y_hat).argmax()

                y_true.extend(labels.cpu().detach().numpy())
                y_pred.append(y_pred_label)

    acc, precision, recall, specificity, f1score = evaluate(y_true, y_pred)
    print("patient-level acc:{:.2f} precision:{} recall:{} specificity:{} f1_score:{}".format(acc, precision, recall, specificity, f1score))
    with open(demo_results_log_path, 'a+') as f:
        f.write('patient-level acc:{:.2f} precision:{} recall:{} specificity:{} f1_score:{}\n\n'.format(acc, precision, recall, specificity, f1score))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Lymph Node Metastasis Classification")
    parser.add_argument("--epochs", type=int, default=30,
                        help="training epochs")
    parser.add_argument("--img_size", type=int, default=224,
                        help="image size")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="initial learning rate")
    parser.add_argument("--log_step", type=int, default=100,
                        help="log accuracy each log_step batchs")
    parser.add_argument("--model_name", type=str, default="resnet",
                        help="model name")
    parser.add_argument("--mode", type=str, default='NAP',
                        help="mode to train or test")
    parser.add_argument("--loc", action="store_true", default=False,
                        help="LNM localization")
    parser.add_argument("--crop", action="store_true", default=False,
                        help="crop lesion location")
    parser.add_argument("--threeD", action="store_true", default=False,
                        help="3D volume")
    parser.add_argument("--dual", action="store_true", default=False,
                        help="dual branch input classification")
    parser.add_argument("--distill", action="store_true", default=False,
                        help="knowledge distillation")
    parser.add_argument("--data_path", type=str, default="data/LNM/dataset/lnm_test_filter.txt",
                        help="data path")
    parser.add_argument("--plt", action="store_true", default=False,
                        help="plot CT image")
    parser.add_argument("--pretrain_weight_path", type=str, default="",
                        help="pretrain weight path")
    parser.add_argument("--model", type=str, default="",
                        help="model to test")
    parser.add_argument("--exp_name", type=str, default="",
                        help="experiment name")

    args = parser.parse_args()

    if args.exp_name:  # Train
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
            train()
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())
            exit(1)

    else:  # test
        test()
