import os
import re
import ast
import cv2
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support

from models.config import CRC_MATCH_PATH


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cal_accuracy(outputs, label):
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == label.data).cpu()

    return corrects


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area

    iou = inter_area / union_area

    return iou


def crop_image(img, bbox, size=80, ori=False):
    scale = 224 / 512
    bbox = [int(x * scale) for x in bbox]
    if ori:
        x1, y1, x2, y2 = bbox
    else:
        square_side = size * scale
        center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        x1, y1, x2, y2 = max(int(center_x - square_side/2), 0), max(int(center_y - square_side/2), 0), \
            min(int(center_x + square_side/2), 224), min(int(center_y + square_side/2), 224)
    img = resize(img[y1:y2+1, x1:x2+1], (224, 224), order=3)

    return img


def evaluate_iou(predicted_boxes, true_boxes, iou_threshold=0.3):
    tp = 0
    total_iou = 0
    for pred_box, true_box in zip(predicted_boxes, true_boxes):
        iou = compute_iou(pred_box, true_box)
        if iou >= iou_threshold:
            tp += 1
            total_iou += iou
    iou = total_iou / tp if tp > 0 else 0
    return iou, tp / len(true_boxes)


def evaluate_boxes(predicted_boxes, y_preds, true_boxes, y_trues, iou_threshold=0):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_iou = 0

    for pred_box, y_pred, true_box, y_true in zip(predicted_boxes, y_preds, true_boxes, y_trues):
        if y_true == 1:
            if y_pred == 1:
                iou = compute_iou(pred_box, true_box)
                if iou >= iou_threshold:
                    true_positives += 1
                    total_iou += iou
                else:
                    false_positives += 1
            else:
                false_negatives += 1
        elif y_pred == 1:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    iou = total_iou / true_positives if true_positives > 0 else 0

    return precision, recall, iou


def evaluate_model(y_true, y_scores, y_pred):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1score = f1_score(y_true, y_pred, zero_division=0)

    return acc, auc, precision, sensitivity, specificity, f1score


def evaluate(y_true, y_pred, output=False):
    def calculate_specificity(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        TN = cm[0, 0]
        FP = cm[0, 1]
        specificity = TN / (TN + FP)
        return specificity

    scale = 100
    accuracy = float('{:.2f}'.format(accuracy_score(y_true, y_pred) * scale))
    precision = float('{:.2f}'.format(precision_score(y_true, y_pred) * scale))
    recall = float('{:.2f}'.format(recall_score(y_true, y_pred) * scale))
    specificity = float('{:.2f}'.format(calculate_specificity(y_true, y_pred) * scale))
    f1 = float('{:.2f}'.format(f1_score(y_true, y_pred) * scale))

    return accuracy, precision, recall, specificity, f1


def test_model_on_dataset(model, data_loader, device, cls_threshold, loc=False):
    total = len(data_loader)
    y_true, y_scores, y_pred, y_true_box, y_pred_box = [], [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            print("{:<5}/{:<5}".format(i+1, total), end="\r")
            inputs, volumes, boxes, labels = data
            inputs = inputs.to(device)
            volumes = volumes.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            if loc:
                outputs, outputs_loc = model(inputs, volumes)
            else:
                outputs = model(inputs, volumes)
            Y_prob = F.softmax(outputs, dim=1)

            Y_hat = (Y_prob[:, 1].cpu().detach().numpy() >= cls_threshold).astype(int)
            y_pred.extend(Y_hat)

            y_true.extend(labels.cpu().detach().numpy())
            y_scores.extend(Y_prob[:, 1].cpu().detach().numpy())

            if loc:
                mask = ~torch.all(boxes == torch.tensor([-1.0, -1.0, -1.0, -1.0]).to(device), dim=1)
                y_true_box.extend(boxes[mask].cpu().detach().numpy())
                y_pred_box.extend(outputs_loc[mask].cpu().detach().numpy())

        print("{:<5}/{:<5}".format(i+1, total))

    acc, auc, precision, sensitivity, specificity, f1score = evaluate_model(y_true, y_scores, y_pred)
    print("acc:{:.2f} auc:{:.2f} precision:{:.2f} sensitivity:{:.2f} specificity:{:.2f} f1_score:{:.2f} ".format(
        acc*100, auc*100, precision*100, sensitivity*100, specificity*100, f1score*100))
    if loc:
        loc_precision, loc_sensitivity, iou = evaluate_boxes(y_pred_box, y_pred, y_true_box, y_true, iou_threshold=0.5)
        print("loc_precision:{:.2f} loc_sensitivity:{:.2f} iou:{:.2f}".format(loc_precision*100, loc_sensitivity*100, iou*100))
    else:
        loc_precision, loc_sensitivity, iou = 0, 0, 0
    return acc, auc, precision, sensitivity, specificity, f1score, loc_precision, loc_sensitivity, iou


def test_model_demo(model, data_loader, threshold, cls_threshold, window_size, crc_test_path, demo_model_name, data_path_type, post, device, loc=False):
    match_dict = json.load(open(CRC_MATCH_PATH, 'r'))
    hard_negatives_dict, hard_positives_dict = {}, {}
    if crc_test_path is not None:
        mode = crc_test_path[0].split('/')[-2]
        cls_post = str(cls_threshold).replace('.', '')
        log_path = f'run/model_soups/{demo_model_name}_{cls_post}_{data_path_type}/{mode}pred_{threshold}_ws_{window_size}_{post}.log'
        if os.path.exists(log_path):
            os.remove(log_path)

    start = 25
    total = len(data_loader)
    y_true, y_pred, y_pred_abnormal, y_true_abnormal, y_hit_abnormal, y_pred_normal = [], [], [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            print("{:<5}/{:<5}".format(i+1, total), end="\r")
            inputs, volumes, labels, label_nums = data
            inputs = inputs.to(device)
            volumes = volumes.to(device)
            labels = labels.to(device)
            label_nums = label_nums.to(device)

            if loc:
                outputs, _ = model(inputs, volumes)
            else:
                outputs = model(inputs, volumes)

            Y_prob = F.softmax(outputs, dim=1)
            Y_hat = (Y_prob[:, 1].cpu().detach().numpy() >= cls_threshold).astype(int)
            y_pred_all = np.array(Y_hat)

            y_filter_pred_all = []
            for k in range(len(y_pred_all)):
                if k < window_size or k > len(y_pred_all) - 1 - window_size:
                    y_filter_pred_all.append(y_pred_all[k])
                else:
                    if np.sum(y_pred_all[k - window_size:k+window_size+1]) > window_size:
                        y_filter_pred_all.append(1)
                    else:
                        y_filter_pred_all.append(0)
            y_filter_pred_all = np.array(y_filter_pred_all)

            y_pred_label = 1 if np.sum(y_filter_pred_all) > threshold else 0

            y_true.extend(labels.cpu().detach().numpy())
            y_pred.append(y_pred_label)

            if crc_test_path is not None:
                if y_true[-1] == 1:
                    y_pred_abnormal.append(y_filter_pred_all)
                    label_num = label_nums.cpu().detach().numpy()[0]
                    y_true_abnormal.append(len(label_num))
                    y_hit_abnormal.append(sum([y_filter_pred_all[idx] for idx in label_num]))

                    y_pred_idx = np.where(y_pred_all == 1)[0]
                    y_pred_idx = [idx + start for idx in y_pred_idx]

                    y_filter_pred_idx = np.where(y_filter_pred_all == 1)[0]
                    y_filter_pred_idx = [idx + start for idx in y_filter_pred_idx]

                    y_true_idx_new = [idx + start for idx in label_num]
                    y_hit_idx = sorted(list(set(y_filter_pred_idx) & set(y_true_idx_new)))

                    with open(log_path, 'a+') as f:
                        print('{} label:{} pred:{} y_true:{} y_pred:{} y_hit:{}\ny_true_idx:{}\ny_pred_idx:{}\ny_filter_pred_idx:{}\ny_hit_idx:{}\n{}'.format(
                            i, y_true[-1], y_pred_label, y_true_abnormal[-1], np.sum(y_filter_pred_all), y_hit_abnormal[-1], y_true_idx_new, y_pred_idx, y_filter_pred_idx, y_hit_idx,
                            os.path.join(match_dict[crc_test_path[i]], 'slice.txt')), file=f)
                        print('all:\n{}\nfilter_all:\n{}'.format(y_pred_all, y_filter_pred_all), file=f)
                        hard_negative, hard_positive = [], []
                        for k in range(len(y_filter_pred_all)):
                            if y_filter_pred_all[k] == 1 and k not in label_num:
                                hard_negative.append(k + start)
                            if y_filter_pred_all[k] == 0 and k in label_num:
                                hard_positive.append(k + start)
                        print('hard_negative:{}\nhard_positive:{}\n'.format(hard_negative, hard_positive), file=f)
                        hard_negatives_dict[crc_test_path[i]] = str(hard_negative)
                        hard_positives_dict[crc_test_path[i]] = str(hard_positive)
                else:
                    y_pred_normal.append(y_filter_pred_all)
                    with open(log_path, 'a+') as f:
                        print('{} label:{} pred:{} y_pred:{} all:{}\n {}'.format(
                            i, y_true[-1], y_pred_label, np.sum(y_filter_pred_all), y_filter_pred_all, crc_test_path[i]), file=f)
                        hard_negative = []
                        for k in range(len(y_filter_pred_all)):
                            if y_filter_pred_all[k] == 1:
                                hard_negative.append(k + start)
                        print('hard_negative:{}\n'.format(hard_negative), file=f)
                        hard_negatives_dict[crc_test_path[i]] = str(hard_negative)
        print("{:<5}/{:<5}".format(i+1, total))

    if crc_test_path is not None:
        hard_negatives_num, hard_positives_num = 0, 0
        for _, v in hard_negatives_dict.items():
            hard_negatives_num += len(v)
        for _, v in hard_positives_dict.items():
            hard_positives_num += len(v)
        with open(log_path, 'a+') as f:
            print('hard negatives:{}, hard positives:{}'.format(hard_negatives_num, hard_positives_num), file=f)

    acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred)

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    f1score = f1_score(y_true, y_pred)
    print("acc:{:.2f} precision:{:.2f} sensitivity:{:.2f} specificity:{:.2f} f1_score:{:.2f}".format(
        acc*100, precision*100, sensitivity*100, specificity*100, f1score*100))
    result = [acc, precision, sensitivity, specificity, f1score]
    y_pred_info = [y_pred_abnormal, y_true_abnormal, y_hit_abnormal, y_pred_normal]

    return result, y_pred_info


def pdf(x):
    mean = np.mean(x)
    std = np.std(x)
    y_out = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
    return y_out


def lcs(x):
    count = 0
    max_count = 0
    for num in x:
        if num == 1:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


def middle_idxs(n_selection, start, end):
    x = np.arange(start, end, 1)
    selection_pdf = pdf(x)
    selection_prob = [float(i) / sum(selection_pdf) for i in selection_pdf]
    replace = False if (end - start) >= n_selection else True
    results = np.random.choice(x, size=n_selection, p=selection_prob, replace=replace)
    return sorted(results)


def uniform_idxs(n_selection, start, end):
    res = []
    while len(res) < n_selection:
        rand_int = int(np.random.uniform(start, end))
        if rand_int not in res:
            res.append(rand_int)
    res = sorted(res)
    return res


def random_idxs(n_selection, start, end):
    x = np.arange(start, end, 1)
    replace = False if (end - start) >= n_selection else True
    res = np.random.choice(x, size=n_selection, replace=replace)
    res = sorted(res)
    return res


def plot_history(train_epoch_acc, train_epoch_auc, train_epoch_precision, train_epoch_sensitivity, train_epoch_specificity, train_epoch_f1score,
                 train_epoch_loc_precision, train_epoch_loc_sensitivity, train_epoch_iou, train_epoch_loss,
                 val_epoch_acc, val_epoch_auc, val_epoch_precision, val_epoch_sensitivity, val_epoch_specificity, val_epoch_f1score,
                 val_epoch_loc_precision, val_epoch_loc_sensitivity, val_epoch_iou, val_epoch_loss, save_path):
    _, ax = plt.subplots(4, 3, figsize=(18, 24))

    ax[0, 0].plot(train_epoch_acc)
    ax[0, 0].plot(val_epoch_acc)
    ax[0, 0].set_title("Model {}".format("ACC"))
    ax[0, 0].set_xlabel("epochs")
    ax[0, 0].legend(["train", "val"])

    ax[0, 1].plot(train_epoch_auc)
    ax[0, 1].plot(val_epoch_auc)
    ax[0, 1].set_title("Model {}".format("AUC"))
    ax[0, 1].set_xlabel("epochs")
    ax[0, 1].legend(["train", "val"])

    ax[0, 2].plot(train_epoch_precision)
    ax[0, 2].plot(val_epoch_precision)
    ax[0, 2].set_title("Model {}".format("Precision"))
    ax[0, 2].set_xlabel("epochs")
    ax[0, 2].legend(["train", "val"])

    ax[1, 0].plot(train_epoch_sensitivity)
    ax[1, 0].plot(val_epoch_sensitivity)
    ax[1, 0].set_title("Model {}".format("Sensitivity"))
    ax[1, 0].set_xlabel("epochs")
    ax[1, 0].legend(["train", "val"])

    ax[1, 1].plot(train_epoch_specificity)
    ax[1, 1].plot(val_epoch_specificity)
    ax[1, 1].set_title("Model {}".format("Specificity"))
    ax[1, 1].set_xlabel("epochs")
    ax[1, 1].legend(["train", "val"])

    ax[1, 2].plot(train_epoch_f1score)
    ax[1, 2].plot(val_epoch_f1score)
    ax[1, 2].set_title("Model {}".format("F1 Score"))
    ax[1, 2].set_xlabel("epochs")
    ax[1, 2].legend(["train", "val"])

    ax[2, 0].plot(train_epoch_loc_precision)
    ax[2, 0].plot(val_epoch_loc_precision)
    ax[2, 0].set_title("Model {}".format("Loc Precision"))
    ax[2, 0].set_xlabel("epochs")
    ax[2, 0].legend(["train", "val"])

    ax[2, 1].plot(train_epoch_loc_sensitivity)
    ax[2, 1].plot(val_epoch_loc_sensitivity)
    ax[2, 1].set_title("Model {}".format("Loc Sensitivity"))
    ax[2, 1].set_xlabel("epochs")
    ax[2, 1].legend(["train", "val"])

    ax[2, 2].plot(train_epoch_iou)
    ax[2, 2].plot(val_epoch_iou)
    ax[2, 2].set_title("Model {}".format("Loc IoU"))
    ax[2, 2].set_xlabel("epochs")
    ax[2, 2].legend(["train", "val"])

    ax[3, 0].plot(train_epoch_loss)
    ax[3, 0].plot(val_epoch_loss)
    ax[3, 0].set_title("Model {}".format("Loss"))
    ax[3, 0].set_xlabel("epochs")
    ax[3, 0].legend(["train", "val"])

    plt.savefig(save_path)
    plt.close()


def plot_crc_threshold(test_result, threshold, m, demo_model_name):
    plt.plot(threshold, test_result[0], label='acc')
    plt.plot(threshold, test_result[1], label='precision')
    plt.plot(threshold, test_result[2], label='sensitivity')
    plt.plot(threshold, test_result[3], label='specificity')
    plt.plot(threshold, test_result[4], label='f1score')

    plt.legend()
    plt.title(f'threshold in CRC model for mode {m}')
    plt.show()
    plt.savefig(f'run/model_soups/{demo_model_name}/threshold{m}.png')
    plt.close()


def plot_predict_info(y_pred_info, m, model_name):
    y_pred_abnormal, y_true_abnormal_info, y_hit_abnormal_info, y_pred_normal = y_pred_info
    y_pred_abnormal_info, y_pred_normal_info = [], []

    for i in range(len(y_pred_abnormal)):
        y_pred_abnormal_info.append(np.sum(y_pred_abnormal[i]))

    for i in range(len(y_pred_normal)):
        y_pred_normal_info.append(np.sum(y_pred_normal[i]))

    fig, ax = plt.subplots(2, 1)
    fig.suptitle(f'patient predict info for mode {m}')

    x1 = np.arange(len(y_true_abnormal_info))
    width = 0.25
    ax[0].bar(x1 - width, y_pred_abnormal_info, width, label='pred')
    ax[0].bar(x1, y_true_abnormal_info, width, label='true')
    ax[0].bar(x1 + width, y_hit_abnormal_info, width, label='overlap')
    ax[0].legend()
    ax[0].set_title('Abnormal')
    ax[0].set_xlabel("patient")
    ax[0].set_ylabel("abnoraml slices")

    x2 = np.arange(len(y_pred_normal_info))
    ax[1].bar(x2, y_pred_normal_info)
    ax[1].set_title('Normal')
    ax[1].set_xlabel("patient")
    ax[1].set_ylabel("abnoraml slices")

    plt.tight_layout()
    plt.show()
    plt.savefig(f'run/model_soups/{model_name}/predict_info{m}_cmp.png')
    plt.close()


def plot_slices(num_rows, num_columns, width, height, scan_mode, data, data_dir=None):
    data = np.transpose(data, (2, 0, 1))
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 6.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            if num_rows == 1:
                axarr[j].text(100, 50, 1 + j + i * columns_data, color='white', fontsize=5)
                axarr[j].imshow(data[i][j], cmap="gray")
                axarr[j].axis("off")
            else:
                axarr[i, j].text(100, 50, 1 + j + i * columns_data, color='white', fontsize=5)
                axarr[i, j].imshow(data[i][j], cmap="gray")
                axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    if data_dir == None:
        plt.savefig("figure/{}CT.png".format(scan_mode), dpi=500)
    else:
        figure_path = os.path.join('figure/{}'.format(data_dir))
        if not os.path.exists(figure_path):
            os.makedirs(figure_path, exist_ok=True)
        plt.savefig("figure/{}/{}CT.png".format(data_dir, scan_mode), dpi=500)


def get_bounding_boxes(heatmap, threshold=0.15, otsu=False):
    """Get bounding boxes from heatmap"""
    p_heatmap = np.array(heatmap*255, np.uint8)
    if otsu:

        threshold, p_heatmap = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:

        p_heatmap[p_heatmap < threshold * 255] = 0
        p_heatmap[p_heatmap >= threshold * 255] = 1

    contours = cv2.findContours(p_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append([x, y, x + w, y + h])

    return bboxes


def get_circle_patches(heatmap, radius=3, color='lightblue'):
    """Get patches for circle point"""
    patches = []
    x = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)[0]
    y = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)[1]

    patches.append(Circle((y, x), radius=radius, color=color))
    return patches


def get_bbox_patches(bboxes, color='r', linewidth=2):
    """Get patches for bounding boxes"""
    patches = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        patches.append(Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', linewidth=linewidth))

    return patches


def create_animation(array, case, box=None, pred_box=None, heatmap=None, alpha=0.3, animation_path=None):
    """Create an animation of images"""
    fig = plt.figure(figsize=(4, 4))
    images = []

    grid_size = 7
    step_size = 224 // grid_size

    for idx, image in enumerate(array):

        image_plot = plt.imshow(image, animated=True, cmap='gray')
        aux = [image_plot]
        if box is not None:

            patches = get_bbox_patches([box[0][idx]], color='red')
            aux.extend(image_plot.axes.add_patch(patch) for patch in patches)

        if pred_box is not None:

            patches = get_bbox_patches([pred_box[idx][0]], color='yellow')
            aux.extend(image_plot.axes.add_patch(patch) for patch in patches)

        if heatmap is not None:
            image_plot2 = plt.imshow(heatmap[idx], animated=True, cmap='jet', alpha=alpha, extent=image_plot.get_extent())
            aux.append(image_plot2)

            patches = get_circle_patches(heatmap[idx])

            bboxes = get_bounding_boxes(heatmap[idx])
            patches = get_bbox_patches(bboxes, color='blue')

        images.append(aux)

    plt.axis('off')
    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, images, interval=3000//len(array), blit=False, repeat_delay=1500)
    plt.close()

    if not os.path.exists(animation_path):
        os.makedirs(animation_path, exist_ok=True)

    ani.save(os.path.join(animation_path, 'animation_{}.gif'.format(case)))

    return ani


def get_region(input):

    class1_probabilities = input[:, 1, :, :]

    _, max_index = torch.max(class1_probabilities.view(-1), 0)

    max_position = (max_index // 7, max_index % 7)

    y, x = max_position[0].item(), max_position[1].item()

    return y, x


def generate_mask(input_tensor):
    batch_size, _, height, width = input_tensor.shape

    mask_tensor = torch.ones_like(input_tensor)

    for b in range(batch_size):

        class_1_probs = input_tensor[b, 1, :, :]

        _, max_index = torch.max(class_1_probs.view(-1), 0)
        max_y, max_x = max_index // 7, max_index % 7
        y1 = max(max_y-1, 0)
        y2 = min(max_y+2, height)
        x1 = max(max_x-1, 0)
        x2 = min(max_x+2, width)

        mask_tensor[b, :, y1:y2, x1:x2] = 0

    return mask_tensor


def fuzzy_search_regex(dictionary, target):
    # Use regular expressions to perform a fuzzy search to check whether the target string is a substring of dictionary keys.
    pattern = re.compile(re.escape(target))
    return [key for key in dictionary.keys() if pattern.search(key)]
