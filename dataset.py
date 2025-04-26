import os
import ast
import cv2
import json
import random
import numpy as np

from scipy import ndimage
from torch.utils.data import Dataset

from utils.utils import random_idxs, crop_image, fuzzy_search_regex
from models.config import MODE_ORDER, SLICE, CRC_MATCH_PATH, CRC_INDEXES_PATH, CRC_ALL_PATH, CRC_BBOX_PATH, CRC_LNM_BBOX_PATH, LABEL_MAP, LABEL_LIST


class CRCDataset(Dataset):
    """crc + 3D scan"""

    def __init__(self, crc_path, mode, transform=None, use_volume=False, align=False, load_indexes_dict=True, type='train'):

        super().__init__()

        self.paths, self.paths3D, self.labels = [], [], []
        self.use_volume = use_volume
        self.align = align
        self.transform = transform

        mode_index = []
        for m in mode:
            mode_index.append(MODE_ORDER.index(m))

        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))

        indexes_dict = {}
        if os.path.exists(CRC_INDEXES_PATH):
            indexes_dict = json.load(open(CRC_INDEXES_PATH, 'r'))

        crc_all_dict = json.load(open(CRC_ALL_PATH, 'r'))

        var_length_abnormal = np.arange(3, 8)
        var_length_normal = np.arange(10, 20)

        crc_path = list(map(list, zip(*crc_path)))

        abnormal_num, box_num, normal_num = 0, 0, 0
        center_list = ['Institution I', 'Institution II', 'Institution III', ]
        center_abnormal = dict.fromkeys(center_list, 0)
        center_normal = dict.fromkeys(center_list, 0)
        for i in mode_index:
            indexes = []
            for path in crc_path[i]:
                center = path.split('/')[1]
                num_slice = int(crc_all_dict[path])

                start, end = 25, num_slice - 12
                label = np.zeros(num_slice)
                if path in indexes_dict:
                    indexes = ast.literal_eval(indexes_dict[path])
                else:
                    abnormal_indexes = []
                    if path in match_dict:
                        num_scans = np.random.choice(var_length_abnormal)
                        for line in open(os.path.join(match_dict[path], 'slice.txt')):
                            abnormal_indexes.append(int(line.split('\t')[0]) - 1)
                    else:
                        num_scans = np.random.choice(var_length_normal)

                    normal_indexes = random_idxs(num_scans, start, end)

                    indexes = sorted(list(set(abnormal_indexes + normal_indexes)))
                    indexes_dict[path] = str(indexes)

                slice_box = {}
                if path in match_dict:
                    abnormal_num += 1
                    center_abnormal[center] += 1
                    for line in open(os.path.join(match_dict[path], 'slice.txt')):
                        slice, gt, bbox = int(line.split('\t')[0]), int(line.split('\t')[2]), ast.literal_eval(line.split('\t')[3].replace('\n', ''))
                        label[slice - 1] = 1
                        slice_box[slice - 1] = bbox
                        if gt == 1:
                            box_num += 1
                else:
                    normal_num += 1
                    center_normal[center] += 1
                for idx in indexes:
                    temp_slice_box = [0, 0, 1.0, 1.0]
                    temp_path3D = [path, idx-SLICE, idx, idx+SLICE]
                    if idx in slice_box:
                        temp_slice_box = [coor/512 for coor in slice_box[idx]]
                        high = idx + SLICE if idx + SLICE < num_slice else idx
                        temp_path3D = [path, idx-1, idx, high]
                    self.paths.append((path, idx, temp_slice_box))
                    self.paths3D.append(temp_path3D)
                    self.labels.append(label[idx])

        if load_indexes_dict:
            with open(CRC_INDEXES_PATH, 'w') as f:
                json.dump(indexes_dict, f, ensure_ascii=False, indent=4)

        self.labels = np.array(self.labels)
        self.num_per_cls_dict = [np.sum(self.labels == 0), np.sum(self.labels == 1)]
        mlen = len(mode_index)
        for c in center_list:
            assert center_abnormal[c] % mlen == 0 and center_normal[c] % mlen == 0
            print(f'{c}: abnormal={int(center_abnormal[c]/mlen)}, normal={int(center_normal[c]/mlen)}')
        print('{}, abnormal patients={} slices={}, boxes={}, normal patients={} slices={}, total patients={} slices={}\n'.format(
            type, int(abnormal_num/mlen), np.sum(self.labels == 1), box_num, int(normal_num/mlen), np.sum(self.labels == 0),
            int((abnormal_num+normal_num)/mlen), np.sum(self.labels == 1)+np.sum(self.labels == 0)))

    def __getitem__(self, idx):

        path = self.paths[idx]
        path3D = self.paths3D[idx]
        label = self.labels[idx]

        scan = np.load(path[0])
        img = scan[:, :, path[1]]

        if self.align:
            scanA = np.load(path[0].replace('N', 'A'))
            scanP = np.load(path[0].replace('N', 'P'))
            try:
                volume = np.stack((scan[:, :, path3D[1]], scan[:, :, path3D[2]], scan[:, :, path3D[3]], scanA[:, :, path3D[2]], scanP[:, :, path3D[2]]),
                                  axis=-1) if self.use_volume else -1
            except:
                volume = np.stack((scan[:, :, path3D[1]], scan[:, :, path3D[2]], scan[:, :, path3D[3]], scan[:, :, path3D[2]], scan[:, :, path3D[2]]),
                                  axis=-1) if self.use_volume else -1
        else:
            volume = np.stack((scan[:, :, path3D[1]], scan[:, :, path3D[2]], scan[:, :, path3D[3]]), axis=-1) if self.use_volume else -1
        box = path[2]

        if self.transform:
            transformed = self.transform(image=img, bboxes=[box], class_labels=[label])
            img = transformed['image']
            box = transformed['bboxes']
            if self.use_volume:
                angles = [-60, -30, -15, 0, 15, 30, 60]
                angle = random.choice(angles)
                volume = ndimage.rotate(volume, angle, reshape=False)
                volume[volume < 0] = 0
                volume[volume > 1] = 1

        img = np.expand_dims(img, axis=0)
        volume = np.expand_dims(volume, axis=0)
        if label == 0 or (self.transform and len(box) == 0):
            box = [(-1.0, -1.0, -1.0, -1.0)]
        if len(box) == 1:
            box = box[0]
        box = np.array(box).astype('float')
        label = np.array(label).astype('int')

        return img, volume, box, label

    def __len__(self):
        return len(self.labels)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(2):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def get_labels(self):
        return self.labels


class CRCPatchDataset(Dataset):
    """crc + Patch Classification"""

    def __init__(self, crc_path, transform=None, type='train'):

        super().__init__()

        self.paths, self.paths3D, self.labels = [], [], []
        self.transform = transform

        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))

        indexes_dict = json.load(open(CRC_INDEXES_PATH, 'r'))

        crc_all_dict = json.load(open(CRC_ALL_PATH, 'r'))

        abnormal_num, box_num, normal_num = 0, 0, 0
        for i in range(len(crc_path)):
            pathN, pathA, pathP = crc_path[i]
            num_slice = int(crc_all_dict[pathN])
            label = np.zeros(num_slice)

            slice_box = {}
            idxN, idxA, idxP = [], [], []
            if pathN in match_dict:
                abnormal_num += 1
                for line in open(os.path.join(match_dict[pathN], 'slice.txt')):
                    slice, gt, bbox = int(line.split('\t')[0]), int(line.split('\t')[2]), ast.literal_eval(line.split('\t')[3].replace('\n', ''))
                    idxN.append(slice - 1)
                    label[slice - 1] = 1
                    slice_box[slice - 1] = bbox
                    if gt == 1:
                        box_num += 1
                for line in open(os.path.join(match_dict[pathA], 'slice.txt')):
                    idxA.append(int(line.split('\t')[0]) - 1)
                for line in open(os.path.join(match_dict[pathP], 'slice.txt')):
                    idxP.append(int(line.split('\t')[0]) - 1)
                idxA = self.map_index(idxN, idxA)
                idxP = self.map_index(idxN, idxP)
                assert len(slice_box) == len(idxA) == len(idxP)
            else:
                normal_num += 1

            k = 0
            indexes = ast.literal_eval(indexes_dict[pathN])
            for idx in indexes:
                temp_slice_box = [0, 0, 1.0, 1.0]
                temp_path3D = [(pathA, idx), idx-1, idx+1, (pathP, idx)]
                if idx in slice_box:
                    temp_slice_box = [coor/512 for coor in slice_box[idx]]
                    high = idx + 1 if idx + 1 < num_slice else idx
                    temp_path3D = [(pathA, idxA[k]), idx-1, high, (pathP, idxP[k])]
                    k += 1
                self.paths.append((pathN, idx, temp_slice_box))
                self.paths3D.append(temp_path3D)
                self.labels.append(label[idx])

        self.labels = np.array(self.labels)
        print('{}, abnormal patients={} slices={}, boxes={}, normal patients={} slices={}'.format(
            type, abnormal_num, np.sum(self.labels == 1), box_num, normal_num, np.sum(self.labels == 0)))

    def __getitem__(self, idx):

        path = self.paths[idx]
        path3D = self.paths3D[idx]

        scanN, scanA, scanP = np.load(path[0]), np.load(path3D[0][0]), np.load(path3D[-1][0])
        img = scanN[:, :, path[1]]
        volume = np.stack((scanA[:, :, path3D[0][1]], scanN[:, :, path3D[1]], scanN[:, :, path3D[2]], scanP[:, :, path3D[-1][1]]), axis=-1)

        box = path[2]
        label = self.labels[idx]

        if self.transform:
            transformed = self.transform(image=img, bboxes=[box], class_labels=[label])
            img = transformed['image']
            box = transformed['bboxes']

            angles = [-60, -30, -15, 0, 15, 30, 60]
            angle = random.choice(angles)
            volume = ndimage.rotate(volume, angle, reshape=False)
            volume[volume < 0] = 0
            volume[volume > 1] = 1

        img = np.expand_dims(img, axis=0)
        volume = np.expand_dims(volume, axis=0)
        if label == 0 or (self.transform and len(box) == 0):
            box = [(-1.0, -1.0, -1.0, -1.0)]
        if len(box) == 1:
            box = box[0]
        box = np.array(box).astype('float')
        label = np.array(label).astype('int')

        return img, volume, box, label

    def __len__(self):
        return len(self.labels)

    def map_index(self, idxA, idx):

        if len(idxA) <= len(idx):
            return idx[:len(idxA)]
        else:
            s = 0
            while len(idxA) > len(idx):
                idx.append(idx[s])
                s += 1
            return sorted(idx)


class CRCDemoDataset(Dataset):
    """patient-level crc"""

    def __init__(self, crc_path, mode='N', use_volume=False, lnm=False, test_grad_cam=False):

        super().__init__()

        self.imgs, self.volumes, self.boxes, self.labels, self.label_nums = [], [], [], [], []

        self.mode_index = []
        for m in mode:
            self.mode_index.append(MODE_ORDER.index(m))
        self.use_volume = use_volume
        self.lnm = lnm
        self.test_grad_cam = test_grad_cam

        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))
        indexes_dict = json.load(open(CRC_INDEXES_PATH, 'r'))
        crc_lnm_bbox = json.load(open(CRC_LNM_BBOX_PATH, 'r'))

        for i in range(len(crc_path)):
            scans, volumes, boxes, indexes_intersection_temp, label_nums = [], [], [], [], []
            label = 1 if crc_path[i][0].find('abnormal') != -1 else 0
            for m_idx in self.mode_index:
                path = crc_path[i][m_idx]
                scan = np.load(path)
                num_slice = scan.shape[2]
                box, label_num = [], []

                if test_grad_cam:
                    if path in match_dict:
                        indexes = []
                        if self.lnm:
                            patient = match_dict[path].split('/')[-2]
                            match_keys = fuzzy_search_regex(crc_lnm_bbox, patient)
                            if len(match_keys) > 0:
                                for match_key in match_keys:
                                    slice = int(match_key.split('_')[-1].split('.')[0])
                                    bbox = crc_lnm_bbox[match_key]
                                    if bbox == [0.0, 0.0, 512.0, 512.0]:
                                        continue
                                    indexes.append(slice - 1)
                                    bbox = [int(b) for b in bbox]
                                    box.append(bbox)
                        else:
                            for line in open(os.path.join(match_dict[path], 'slice.txt')):
                                slice, bbox, gt = line.split('\t')[0], ast.literal_eval(line.split('\t')[1]), line.split('\t')[2]
                                if int(gt) == 1:
                                    indexes.append(int(slice) - 1)
                                    box.append(bbox)
                        sorted_lists = sorted(zip(indexes, box), key=lambda pair: pair[0])
                        indexes = [x[0] for x in sorted_lists]
                        box = [x[1] for x in sorted_lists]

                    else:

                        indexes = ast.literal_eval(indexes_dict[path])
                        box = []
                        for _ in range(len(indexes)):
                            box.append([0, 0, 0, 0])

                else:
                    start, end = 25, num_slice - 12
                    indexes = np.arange(start, end, 1)
                    if path in match_dict:
                        for line in open(os.path.join(match_dict[path], 'slice.txt')):
                            slice = line.split('\t')[0]

                            if int(slice) - 1 - start < 0 or int(slice) > end:
                                continue
                            label_num.append(int(slice) - 1 - start)
                    else:
                        label_num.append(0)
                    label_nums.append(label_num)

                if self.use_volume:
                    volume = []
                    for idx in indexes:
                        low = idx - SLICE
                        high = idx + SLICE
                        if high + 1 > num_slice:
                            stacked_scan = np.stack([scan[:, :, -1]] * (high + 1 - num_slice), axis=-1)
                            volume.append(np.concatenate((scan[:, :, low:num_slice], stacked_scan), axis=-1))
                        else:
                            volume.append(np.array(scan[:, :, low:high+1]))
                    volumes.append(volume)

                scan = [scan[:, :, idx] for idx in indexes]
                scans.append(scan)
                boxes.append(box)

            if len(self.mode_index) == 3 and not test_grad_cam:
                scans = np.concatenate((scans[0], scans[1], scans[2]), axis=0)
                if self.use_volume:
                    volumes = np.concatenate((volumes[0], volumes[1], volumes[2]), axis=0)
                label_nums = np.concatenate((label_nums[0], label_nums[1], label_nums[2]), axis=0)

            self.imgs.append(scans)
            if self.use_volume:
                self.volumes.append(volumes)
            self.boxes.append(boxes)
            self.labels.append(label)
            self.label_nums.append(label_nums)

        self.labels = np.array(self.labels)
        print('abnormal: {}, normal: {}'.format(np.sum(self.labels == 1), np.sum(self.labels == 0)))

    def __getitem__(self, idx):
        img = self.imgs[idx][0] if len(self.mode_index) == 1 else self.imgs[idx]
        if self.use_volume:
            volume = self.volumes[idx][0] if len(self.mode_index) == 1 else self.volumes[idx]
        else:
            volume = -1
        label = self.labels[idx]

        if self.test_grad_cam:
            box = self.boxes[idx]
            mask = []
            for i in range(len(img)):
                img[i] = np.expand_dims(img[i], axis=1)
                if self.use_volume:
                    volume[i] = np.expand_dims(volume[i], axis=1)

                if box[i][0] is not None:
                    temp_mask = []
                    for bbox in box[i]:
                        _mask = np.zeros((512, 512))
                        _mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = 1
                        _mask = np.array(cv2.resize(_mask, (224, 224)))
                        temp_mask.append(_mask)
                    mask.append(temp_mask)
                    box[i] = np.array(box[i])*(224/512)

            return img, volume, box, label, mask
        else:
            label_num = self.label_nums[idx][0]
            img = np.expand_dims(img, axis=1)
            if self.use_volume:
                volume = np.expand_dims(volume, axis=1)
            else:
                volume = np.expand_dims(volume, axis=0)
            label_num = np.array(label_num)
            return img, volume, label, label_num

    def __len__(self):
        return len(self.labels)


class LNMDataset(Dataset):
    """lnm dataset"""

    def __init__(self, lnm_path, mode, transform=None, crop=False, loc=False, type='train'):

        super().__init__()

        self.paths, self.bboxs, self.labels = [], [], []
        self.transform = transform
        self.crop = crop
        self.loc = loc

        mode_index = []
        for m in mode:
            mode_index.append(MODE_ORDER.index(m))

        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))

        indexes_dict = json.load(open(CRC_INDEXES_PATH, 'r'))

        crc_lnm_bbox = json.load(open(CRC_LNM_BBOX_PATH, 'r'))

        lnm_path = list(map(list, zip(*lnm_path)))

        lnm = dict.fromkeys(LABEL_LIST, 0)
        for i in mode_index:
            for path, label in zip(lnm_path[i], lnm_path[3]):
                label = LABEL_MAP[int(label)]
                if len(LABEL_LIST) == 2 and label == 2:
                    continue
                lnm[label] += 1
                if path in match_dict:

                    if self.loc or self.crop:
                        patient = match_dict[path].split('/')[-2]
                        match_keys = fuzzy_search_regex(crc_lnm_bbox, patient)
                        if len(match_keys) > 0:
                            for match_key in match_keys:
                                slice = int(match_key.split('_')[-1].split('.')[0])
                                bbox = crc_lnm_bbox[match_key]
                                if bbox == [0.0, 0.0, 512.0, 512.0]:
                                    continue
                                self.paths.append((path, slice - 1, match_dict[path]))
                                self.bboxs.append(bbox)
                                self.labels.append(label)
                        else:
                            lnm[label] -= 1

                    else:
                        for line in open(os.path.join(match_dict[path], 'slice.txt')):
                            slice, bbox = int(line.split('\t')[0]), ast.literal_eval(line.split('\t')[3].replace('\n', ''))
                            self.paths.append((path, slice - 1, match_dict[path]))
                            self.bboxs.append(bbox)
                            self.labels.append(label)
                else:
                    indexes = ast.literal_eval(indexes_dict[path])
                    for idx in indexes:
                        self.paths.append((path, idx, -1))
                        self.bboxs.append(None)
                        self.labels.append(label)
        self.labels = np.array(self.labels)

        mlen = len(mode_index)
        lnm = {key: value // mlen for key, value in lnm.items()}
        print(lnm)
        print('{} patient LNM:{} {}, slice N0:{} N1:{} N2:{}'.format(
            type, sum(lnm.values()), lnm, np.sum(self.labels == 0), np.sum(self.labels == 1), np.sum(self.labels == 2)))

    def __getitem__(self, idx):
        path = self.paths[idx]
        bbox = self.bboxs[idx]

        scan = np.load(path[0])
        img = scan[:, :, path[1]]
        label = self.labels[idx]

        if self.crop and len(LABEL_LIST) == 2:
            img = crop_image(img, bbox, ori=False)

        if self.loc:
            bbox = [coor / 512 for coor in bbox]

        if self.transform:

            if self.loc:
                transformed = self.transform(image=img, bboxes=[bbox], class_labels=[label])
                img = transformed['image']
                bbox = transformed['bboxes']

            else:
                img = self.transform(image=img)["image"]

        img = np.expand_dims(img, axis=0)
        if self.loc:
            if len(bbox) == 1:
                bbox = bbox[0]
            elif len(bbox) == 0:
                return self.__getitem__((idx + 1) % len(self.labels))
            bbox = np.array(bbox).astype('float32')
            return img, bbox
        else:
            label = np.array(label).astype('int')
            return img, label

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels


class LNMDualDataset(Dataset):
    """crc + lnm dataset"""

    def __init__(self, lnm_path, mode, transform=None, type='train'):

        super().__init__()

        self.paths, self.bboxs, self.labels = [], [], []
        self.transform = transform

        temp_lnm_path = []

        mode_index = []
        for m in mode:
            mode_index.append(MODE_ORDER.index(m))

        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))

        crc_bbox = json.load(open(CRC_BBOX_PATH, 'r'))
        crc_lnm_bbox = json.load(open(CRC_LNM_BBOX_PATH, 'r'))

        lnm_path = list(map(list, zip(*lnm_path)))

        lnm = dict.fromkeys(LABEL_LIST, 0)
        for i in mode_index:
            for path, label in zip(lnm_path[i], lnm_path[3]):
                ori_label = label
                label = LABEL_MAP[int(label)]
                if len(LABEL_LIST) == 2 and label == 2:
                    continue

                if path in match_dict:

                    patient = match_dict[path].split('/')[-2]
                    match_keys = fuzzy_search_regex(crc_lnm_bbox, patient)
                    match_keys_full = fuzzy_search_regex(crc_bbox, patient)
                    if len(match_keys) > 0:

                        for match_key in match_keys:
                            for match_key_full in match_keys_full:
                                slice_full = int(match_key_full.split('_')[-1].split('.')[0])
                                slice = int(match_key.split('_')[-1].split('.')[0])
                                if abs(slice_full - slice) > 1 or (len(match_keys) == 1 and slice_full != slice):
                                    continue

                                bbox = crc_bbox[match_key_full]
                                lnm_bbox = crc_lnm_bbox[match_key]

                                self.paths.append((path, slice_full - 1, slice - 1))
                                self.bboxs.append((bbox, lnm_bbox))
                                self.labels.append(label)
                        lnm[label] += 1

                        temp_lnm_path.append([path.replace('P', 'N'), path.replace('P', 'A'), path, ori_label])

        self.labels = np.array(self.labels)

        mlen = len(mode_index)
        lnm = {key: value // mlen for key, value in lnm.items()}
        print(lnm)
        print('{} patient LNM:{} {}, slice N0:{} N1:{} N2:{}'.format(
            type, sum(lnm.values()), lnm, np.sum(self.labels == 0), np.sum(self.labels == 1), np.sum(self.labels == 2)))

    def __getitem__(self, idx):
        path = self.paths[idx]
        bbox = self.bboxs[idx]

        scan = np.load(path[0])
        img_crc = scan[:, :, path[1]]
        img_lnm = scan[:, :, path[2]]
        label = self.labels[idx]

        img1 = crop_image(np.copy(img_crc), bbox[0], size=200, ori=False)
        img2 = crop_image(np.copy(img_lnm), bbox[1], size=80, ori=False)

        if self.transform:

            img1 = self.transform(image=img1)["image"]
            img2 = self.transform(image=img1)["image"]

        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        img = np.stack([img1, img2], axis=0)
        label = np.array(label).astype('int')

        return img, label

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels


class LNMDualDemoDataset(Dataset):
    """patient-level crc + lnm dataset"""

    def __init__(self, lnm_path, mode):

        super().__init__()

        self.paths, self.bboxs, self.labels = [], [], []

        mode_index = []
        for m in mode:
            mode_index.append(MODE_ORDER.index(m))

        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))

        crc_bbox = json.load(open(CRC_BBOX_PATH, 'r'))
        crc_lnm_bbox = json.load(open(CRC_LNM_BBOX_PATH, 'r'))

        lnm = dict.fromkeys(LABEL_LIST, 0)
        lnm_jh = dict.fromkeys(LABEL_LIST, 0)
        lnm_lhl = dict.fromkeys(LABEL_LIST, 0)
        for i in range(len(lnm_path)):
            label = LABEL_MAP[int(lnm_path[i][3])]
            if len(LABEL_LIST) == 2 and label == 2:
                continue

            paths, bboxs = [], []
            for m_idx in mode_index:
                path = lnm_path[i][m_idx]
                if path in match_dict:

                    patient = match_dict[path].split('/')[-2]
                    match_keys = fuzzy_search_regex(crc_lnm_bbox, patient)
                    match_keys_full = fuzzy_search_regex(crc_bbox, patient)
                    if len(match_keys) > 0:

                        for match_key in match_keys:
                            for match_key_full in match_keys_full:
                                slice_full = int(match_key_full.split('_')[-1].split('.')[0])
                                slice = int(match_key.split('_')[-1].split('.')[0])
                                if abs(slice_full - slice) > 1 or (len(match_keys) == 1 and slice_full != slice):
                                    continue
                                bbox = crc_bbox[match_key_full]
                                lnm_bbox = crc_lnm_bbox[match_key]
                                paths.append((path, slice_full - 1, slice - 1))
                                bboxs.append((bbox, lnm_bbox))
                        lnm[label] += 1

                        center = path.split('/')[1]
                        if center == 'Institution III':
                            lnm_jh[label] += 1
                        elif center == 'Institution II':
                            lnm_lhl[label] += 1

            if len(paths) != 0:
                self.paths.append(paths)
                self.bboxs.append(bboxs)
                self.labels.append(label)

        print('patient LNM:{} {}'.format(sum(lnm.values()), lnm))
        print('patient jinhua LNM:{} {}'.format(sum(lnm_jh.values()), lnm_jh))
        print('patient lhl LNM:{} {}'.format(sum(lnm_lhl.values()), lnm_lhl))

    def __getitem__(self, idx):
        paths = self.paths[idx]
        bboxs = self.bboxs[idx]
        label = self.labels[idx]

        imgs1, imgs2 = [], []
        for i, path in enumerate(paths):
            scan = np.load(path[0])
            img_crc = scan[:, :, path[1]]
            img_lnm = scan[:, :, path[2]]

            img1 = crop_image(np.copy(img_crc), bboxs[i][0], size=200, ori=False)
            imgs1.append(np.expand_dims(img1, axis=0))

            img2 = crop_image(np.copy(img_lnm), bboxs[i][1], size=80, ori=False)
            imgs2.append(np.expand_dims(img2, axis=0))

        imgs1 = np.array(imgs1)
        imgs2 = np.array(imgs2)

        img = np.expand_dims(np.concatenate((imgs1, imgs2), axis=1), axis=2)

        return img, label

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels


class LNMDemoDataset(Dataset):
    """patient-level lnm dataset"""

    def __init__(self, lnm_path, mode, crop=False):

        super().__init__()

        self.paths, self.bboxs, self.labels = [], [], []
        self.crop = crop

        mode_index = []
        for m in mode:
            mode_index.append(MODE_ORDER.index(m))

        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))

        crc_lnm_bbox = json.load(open(CRC_LNM_BBOX_PATH, 'r'))

        lnm = dict.fromkeys(LABEL_LIST, 0)
        for i in range(len(lnm_path)):
            label = LABEL_MAP[int(lnm_path[i][3])]
            if len(LABEL_LIST) == 2 and label == 2:
                continue

            paths, bboxs = [], []
            for m_idx in mode_index:
                path = lnm_path[i][m_idx]
                if path in match_dict:

                    patient = match_dict[path].split('/')[-2]
                    match_keys = fuzzy_search_regex(crc_lnm_bbox, patient)
                    if len(match_keys) > 0:
                        for match_key in match_keys:
                            slice = int(match_key.split('_')[-1].split('.')[0])
                            lnm_bbox = crc_lnm_bbox[match_key]
                            paths.append((path, slice - 1))
                            bboxs.append(lnm_bbox)
                        lnm[label] += 1

            if len(paths) != 0:
                self.paths.append(paths)
                self.bboxs.append(bboxs)
                self.labels.append(label)

        print('patient LNM:{} {}'.format(sum(lnm.values()), lnm))

    def __getitem__(self, idx):
        paths = self.paths[idx]
        bboxs = self.bboxs[idx]
        label = self.labels[idx]

        assert len(paths) == len(bboxs)

        imgs = []
        for i, path in enumerate(paths):
            scan = np.load(path[0])
            img = scan[:, :, path[1]]
            if self.crop:
                img = crop_image(img, bboxs[i], ori=False)
            imgs.append(np.expand_dims(img, axis=0))

        imgs = np.array(imgs)

        return imgs, label

    def __len__(self):
        return len(self.labels)


class LNM3DDataset(Dataset):
    """patient-level lnm 3d dataset"""

    def __init__(self, lnm_path, mode, crop=False, crc_all=False, type='train'):

        super().__init__()

        self.paths, self.labels = [], []
        self.crop = crop
        self.type = type
        self.crc_all = crc_all

        match_dict = json.load(open(CRC_MATCH_PATH, 'r'))

        crc_all_dict = json.load(open(CRC_ALL_PATH, 'r'))

        mode_index = []
        for m in mode:
            mode_index.append(MODE_ORDER.index(m))

        lnm = dict.fromkeys(LABEL_LIST, 0)
        for i in range(len(lnm_path)):
            label = LABEL_MAP[int(lnm_path[i][3])]
            if len(LABEL_LIST) == 2 and label == 2:
                continue
            lnm[label] += 1
            temp_paths = []

            for m_idx in mode_index:
                path = lnm_path[i][m_idx]
                indexes, bboxs = [], []
                if self.crc_all:
                    num_slice = crc_all_dict[path]
                    start, end = 25, num_slice - 12
                    indexes = np.arange(start, end, 1)
                else:
                    for line in open(os.path.join(match_dict[path], 'slice.txt')):
                        slice, bbox = int(line.split('\t')[0]), ast.literal_eval(line.split('\t')[3].replace('\n', ''))
                        indexes.append(slice - 1)
                        bboxs.append(bbox)
                temp_paths.append((path, indexes, bboxs))
            self.paths.append(temp_paths)
            self.labels.append(label)

        print('patient LNM:{} {}'.format(sum(lnm.values()), lnm))

    def __getitem__(self, idx):
        paths = self.paths[idx]
        label = self.labels[idx]
        scans = []
        for item in paths:

            scan = np.load(item[0])
            if self.crop:
                scan = [crop_image(scan[:, :, idx], item[2][k], ori=False) for k, idx in enumerate(item[1])]
            else:
                scan = [scan[:, :, idx] for idx in item[1]]
            scan = np.array(scan).transpose(1, 2, 0)
            scans.append(scan)
        if len(scans) == 3:
            new_dim = 24
            scans = np.concatenate((scans[0], scans[1], scans[2]), axis=2)
        else:
            new_dim = 64 if self.crc_all else 8
            scans = np.array(scans[0])
        zoom_factors = [1, 1, new_dim / scans.shape[-1]]
        scans = ndimage.zoom(scans, zoom_factors, order=1)

        if self.type == 'train':
            scans = self.augmentation(scans)

        scans = np.expand_dims(scans, axis=0)

        return scans, label

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def augmentation(self, scan):
        angles = [-60, -30, -15, 0, 15, 30, 60]
        angle = random.choice(angles)
        scan = ndimage.rotate(scan, angle, reshape=False)

        scan[scan < 0] = 0
        scan[scan > 1] = 1

        return scan
