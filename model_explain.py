import os
import ast
import cv2
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn

from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.ndimage import zoom

from dataset import CRCDemoDataset
from models.crc_model import CRCModel
from models.config import MODE_ORDER
from utils.utils import set_seed, create_animation, get_region, get_bounding_boxes


set_seed(seed=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size", type=int, default=224, help="image size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
    )
    parser.add_argument(
        "--use_volume", action="store_true", default=False, help="use volume"
    )
    parser.add_argument(
        "--p_cls", action="store_true", default=False, help="patch cls"
    )
    parser.add_argument(
        "--loc", action="store_true", default=False, help="lesion location"
    )
    parser.add_argument(
        "--lnm", action="store_true", default=False, help="lnm visualization"
    )
    parser.add_argument(
        "--vis", action="store_true", default=False, help="visualization"
    )
    parser.add_argument(
        "--pretrain_weight_path", type=str, default="", help="pretrain weight path"
    )

    return parser.parse_args()


def get_hitrate(gt_mask, heatmap):
    """Pointing game """
    hit = 0
    x = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)[0]
    y = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)[1]
    if gt_mask[x][y] == 1:
        hit = 1
    return hit


def calculate_mask_iou(gt_mask, pred_mask):
    """
    Calculate IoU score between two masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    if np.sum(union) == 0:
        iou_score = 0
    else:
        iou_score = np.sum(intersection) / (np.sum(union))

    return iou_score


def calculate_mask_dice(gt_mask, pred_mask):
    dice = 2*(gt_mask*pred_mask).sum() / (gt_mask.sum()+pred_mask.sum())

    return dice


def get_grid_matrix(box):
    image_size = (224, 224)
    grid_size = (7, 7)
    cell_size = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])

    matrix = np.zeros(grid_size)

    xmin, ymin, xmax, ymax = box

    grid_xmin = int(xmin // cell_size[0])
    grid_ymin = int(ymin // cell_size[1])
    grid_xmax = min(int(xmax // cell_size[0]), 6)
    grid_ymax = min(int(ymax // cell_size[1]), 6)

    for x in range(grid_xmin, grid_xmax + 1):
        for y in range(grid_ymin, grid_ymax + 1):
            matrix[y, x] = 1

    return matrix


def eval_loc():
    # locating
    TP = 0
    FP = 0
    FN = 0

    for i, data in enumerate(test_loader, 0):
        print("{:<5}/{:<5}".format(i+1, total), end="\r")
        inputs, _, boxes, labels, _ = data
        labels = labels.to(device)
        for j in range(len(inputs)):  # N mode
            if j > 0:
                break
            image, y_preds, y_trues, pred_boxes = [], [], [], []
            for k in range(len(inputs[j][0])):
                input_tensor = inputs[j][0][k].unsqueeze(0).to(device)  # image
                image.append(np.squeeze(input_tensor.cpu().detach().numpy()))
                y_trues.append(labels.cpu().detach().numpy())

                if args.loc:  # loc
                    outputs, outputs_loc = model(input_tensor)
                    temp_box = outputs_loc.cpu().detach().numpy() * 224
                    pred_boxes.append(temp_box)  # fix, box times 224

                else:  # p_cls
                    outputs, outputs_p_cls = model(input_tensor)

                    max_y, max_x = get_region(outputs_p_cls)
                    nei0 = 0.5
                    nei1 = 2 - nei0
                    pred_boxes.append(np.array([[(max_y+nei0)*32, (max_x+nei0)*32, min((max_y+nei1)*32, 224-1), min((max_x+nei1)*32, 224-1)]]))

                y_preds.append(torch.argmax(outputs, dim=1).cpu().detach().numpy())

            for pred_box, true_box in zip(pred_boxes,  boxes[j][0].cpu().detach().numpy()):
                # grid matrix
                pred_grid = get_grid_matrix(pred_box[0])
                gt_grid = get_grid_matrix(true_box)

                # print('pred_grid:\n', pred_grid)
                # print('gt_grid:\n', gt_grid)

                TP += np.sum(np.logical_and(pred_grid == 1, gt_grid == 1))
                FP += np.sum(np.logical_and(pred_grid == 1, gt_grid == 0))
                FN += np.sum(np.logical_and(pred_grid == 0, gt_grid == 1))

            if y_trues[-1] == 1 and args.vis:
                if args.loc:
                    create_animation(image, f'{i+1}{MODE_ORDER[j]}loc', boxes[j].cpu().detach().numpy(),
                                     pred_boxes, None, animation_path=animation_path)
                else:
                    create_animation(image, f'{i+1}{MODE_ORDER[j]}mt', boxes[j].cpu().detach().numpy(),
                                     pred_boxes, None, animation_path=animation_path)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    iou = TP / (TP + FP + FN)
    dice = 2 * TP / (2 * TP + FP + FN)
    print('Grid evaluation')
    print('precision:{:.2f} recall:{:.2f} iou:{:.2f} dice:{:.2f}'.format(precision*100, recall*100, iou*100, dice*100))


def eval_cam():
    """eval cam"""
    target_layers = [model.module.model.layer4[-1]]

    cam = EigenCAM(model=model, target_layers=target_layers)

    TP = 0
    FP = 0
    FN = 0

    for i, data in enumerate(test_loader, 0):
        print("{:<5}/{:<5}".format(i+1, total), end="\r")
        inputs, _, boxes, labels, input_masks = data
        labels = labels.to(device)
        images, heatmaps, masks = [], [], []
        y_trues, y_preds = [], []
        for j in range(len(inputs)):  # NAP mode
            if j > 0:
                break
            image, heatmap, mask = [], [], []
            y_true, y_pred = [], []
            for k in range(len(inputs[j][0])):
                input_tensor = inputs[j][0][k].unsqueeze(0).to(device)
                # GradCAM
                targets = [ClassifierOutputTarget(labels)]
                grayscale_cam = cam(input_tensor, targets)
                grayscale_cam = grayscale_cam[0, :]
                image.append(np.squeeze(input_tensor.cpu().detach().numpy()))

                if args.vis:
                    grayscale_cam[grayscale_cam <= args.threshold] = 0  # shape:(224, 224)
                else:
                    grayscale_cam = (grayscale_cam >= args.threshold).astype(int)

                y_true.append(labels.cpu().detach().numpy()[0])
                y_pred.append(torch.argmax(cam.outputs, dim=1).cpu().detach().numpy()[0])

                heatmap.append(grayscale_cam)
            for input_mask in input_masks[j]:  # mask
                mask.append(np.squeeze(input_mask.cpu().detach().numpy()))

            images.append(image)
            masks.append(mask)
            heatmaps.append(heatmap)
            y_trues.append(y_true)
            y_preds.append(y_pred)

            if args.vis:
                create_animation(image, f'{i+1}{MODE_ORDER[j]}heatmap', box=boxes[j].cpu().detach().numpy(), pred_box=None,
                                 heatmap=heatmap, animation_path=animation_path)

        # evaluate
        for j in range(len(heatmaps)):
            for k in range(len(heatmaps[j])):
                pred_box = get_bounding_boxes(heatmaps[j][k])
                true_box = get_bounding_boxes(masks[j][k])

                pred_grid = get_grid_matrix(pred_box[0])
                gt_grid = get_grid_matrix(true_box[0])

                TP += np.sum(np.logical_and(pred_grid == 1, gt_grid == 1))
                FP += np.sum(np.logical_and(pred_grid == 1, gt_grid == 0))
                FN += np.sum(np.logical_and(pred_grid == 0, gt_grid == 1))

    print("{:<5}/{:<5}\n".format(i+1, total))

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    iou = TP / (TP + FP + FN)
    dice_coefficient = 2 * TP / (2 * TP + FP + FN)
    print('precision:{:.2f} recall:{:.2f} iou:{:.2f} dice:{:.2f}'.format(precision*100, recall*100, iou*100, dice_coefficient*100))


def eval_lnm_cam():
    # resnet50
    target_layers = [model.module.model.layer4[-1]]

    cam = EigenCAM(model=model, target_layers=target_layers)

    TP = 0
    FP = 0
    FN = 0

    for i, data in enumerate(test_loader, 0):
        print("{:<5}/{:<5}".format(i+1, total), end="\r")
        inputs, _, boxes, labels, input_masks = data
        labels = labels.to(device)
        images, heatmaps, masks = [], [], []
        y_trues, y_preds = [], []
        for j in range(len(inputs)):  # NAP mode
            if j != 2:
                continue
            image, heatmap, mask = [], [], []
            y_true, y_pred = [], []
            for k in range(len(inputs[j][0])):
                input_tensor = inputs[j][0][k].unsqueeze(0).to(device)
                # GradCAM
                targets = [ClassifierOutputTarget(labels)]
                grayscale_cam = cam(input_tensor, targets)
                grayscale_cam = grayscale_cam[0, :]
                image.append(np.squeeze(input_tensor.cpu().detach().numpy()))

                if args.vis:
                    grayscale_cam[grayscale_cam <= args.threshold] = 0  # shape:(224, 224)
                else:
                    grayscale_cam = (grayscale_cam >= args.threshold).astype(int)

                y_true.append(labels.cpu().detach().numpy()[0])
                y_pred.append(torch.argmax(cam.outputs, dim=1).cpu().detach().numpy()[0])

                heatmap.append(grayscale_cam)
            for input_mask in input_masks[j]:  # mask
                mask.append(np.squeeze(input_mask.cpu().detach().numpy()))

            images.append(image)
            masks.append(mask)
            heatmaps.append(heatmap)
            y_trues.append(y_true)
            y_preds.append(y_pred)

            if args.vis:
                create_animation(image, f'{i+1}{MODE_ORDER[j]}heatmap', box=boxes[j].cpu().detach().numpy(), pred_box=None,
                                 heatmap=heatmap, animation_path=animation_path)

        # evaluate
        for j in range(len(heatmaps)):
            for k in range(len(heatmaps[j])):
                pred_box = get_bounding_boxes(heatmaps[j][k])
                true_box = get_bounding_boxes(masks[j][k])

                pred_grid = get_grid_matrix(pred_box[0])
                gt_grid = get_grid_matrix(true_box[0])

                TP += np.sum(np.logical_and(pred_grid == 1, gt_grid == 1))
                FP += np.sum(np.logical_and(pred_grid == 1, gt_grid == 0))
                FN += np.sum(np.logical_and(pred_grid == 0, gt_grid == 1))

    print("{:<5}/{:<5}\n".format(i+1, total))

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    iou = TP / (TP + FP + FN)
    dice_coefficient = 2 * TP / (2 * TP + FP + FN)
    print('precision:{:.2f} recall:{:.2f} iou:{:.2f} dice:{:.2f}'.format(precision*100, recall*100, iou*100, dice_coefficient*100))


if __name__ == '__main__':
    args = parse_arguments()

    crc_path = []
    for line in open('data/EOLO/crc_eo_val.txt'):  # crc
        temp_path = ast.literal_eval(line.strip())
        if temp_path[0].find('abnormal') != -1:
            crc_path.append(ast.literal_eval(line.strip()))  # abnormal

    test_dataset = CRCDemoDataset(crc_path, mode='NAP', use_volume=args.use_volume, lnm=args.lnm, test_grad_cam=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24, pin_memory=True)
    total = len(test_loader)
    print("Total patients: {}".format(len(test_loader.dataset)))

    model = CRCModel(model_name='resnet', loc=args.loc, test_grad_cam=True)
    model = nn.DataParallel(model)

    model_name = args.pretrain_weight_path.split('/')[-2]
    model.load_state_dict(torch.load(args.pretrain_weight_path), strict=False)

    model.to(device)
    model.eval()

    vis = 'vis' if args.vis else ''
    if args.p_cls:
        animation_path = os.path.join('run', 'vis', f'{model_name}_mt_{vis}')
        eval_loc()
    elif args.loc:
        animation_path = os.path.join('run', 'vis', f'{model_name}_loc_{vis}')
        eval_loc()
    else:
        animation_path = os.path.join('run', 'vis', f'{model_name}_cam_{vis}')
        if args.lnm:
            eval_lnm_cam()
        else:
            eval_cam()
