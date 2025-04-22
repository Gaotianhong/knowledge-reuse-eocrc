import os
import ast
import json
import torch
import argparse

import numpy as np
import torch.nn as nn

from dataset import CRCDataset, CRCDemoDataset
from models.crc_model import CRCModel
from models.config import MODE_ORDER
from utils.utils import set_seed, test_model_on_dataset, test_model_demo, plot_crc_threshold, plot_predict_info


set_seed(seed=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size", type=int, default=224, help="image size"
    )
    parser.add_argument(
        "--model_name", type=str, default="convnextv2_large.fcmae_ft_in22k_in1k", help="model name"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="N_student_EO_loc_3D", help="ckpt path"
    )
    parser.add_argument(
        "--data_path", type=str, default="data/EOLO_new/crc_eo_test.txt", help="data path"
    )
    parser.add_argument(
        "--mode", type=str, default='N', help="mode to test"
    )
    parser.add_argument(
        "--use_volume", action="store_true", default=False, help="use volume"
    )
    parser.add_argument(
        "--loc", action="store_true", default=False, help="lesion localization"
    )
    parser.add_argument(
        "--eval_individual_models", action="store_true", default=False,
    )
    parser.add_argument(
        "--uniform_soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--greedy_soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
    )

    return parser.parse_args()


def get_evaluate_results(model, data_loader, cls_threshold, model_name, m=''):
    results = {'model_name': model_name}
    acc, auc, precision, sensitivity, specificity, f1score, loc_precision, loc_sensitivity, iou = test_model_on_dataset(model, data_loader, device, cls_threshold, args.loc)
    if m == '':
        results['acc'] = acc
        results['auc'] = auc
        results['precision'] = precision
        results['sensitivity'] = sensitivity
        results['specificity'] = specificity
        results['f1score'] = f1score
        results['loc_precision'] = loc_precision
        results['loc_sensitivity'] = loc_sensitivity
        results['iou'] = iou
    else:
        results[f'{m}_acc'] = acc
        results[f'{m}_auc'] = auc
        results[f'{m}_precision'] = precision
        results[f'{m}_sensitivity'] = sensitivity
        results[f'{m}_specificity'] = specificity
        results[f'{m}_f1score'] = f1score
        results[f'{m}_loc_precision'] = loc_precision
        results[f'{m}_loc_sensitivity'] = loc_sensitivity
        results[f'{m}_iou'] = iou

    return results


def evaluate():
    data_path_type = args.data_path.split('/')[-1].split('.')[0]
    demo_save_path = os.path.join('run', 'model_soups', demo_model_name + '_' + cls_post + '_' + data_path_type)
    if not os.path.exists(demo_save_path):
        os.makedirs(demo_save_path, exist_ok=True)

    post = 'img'  # img
    if args.use_volume:
        post = 'img_volume'  # img + volume
    s = 100

    model_name = demo_model_name + '_' + model_path.split('/')[-1].split('.')[0]
    demo_results_log_path = f'run/model_soups/{demo_model_name}_{cls_post}_{data_path_type}/demo_results_{post}.log'
    if os.path.exists(demo_results_log_path):
        os.remove(demo_results_log_path)
    print(model_name)

    if len(mode_index) == 3:
        results = get_evaluate_results(model, test_loader, cls_threshold, model_name)
        with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')
        with open(demo_results_log_path, 'a+') as f:
            f.write('model_name:{}\nacc:{:.2f} auc:{:.2f} precision:{:.2f} sensitivity:{:.2f} specificity:{:.2f} f1score:{:.2f}\n'.format(
                results['model_name'], results['acc']*s, results['auc']*s, results['precision']*s, results['sensitivity']*s, results['specificity']*s, results['f1score']*s))

    for i in mode_index:
        m = MODE_ORDER[i]
        print(f'mode {m}')
        results = get_evaluate_results(model, test_loader_single_mode[0], cls_threshold, model_name + '_' + m, m)
        with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')
        with open(demo_results_log_path, 'a+') as f:
            f.write('model_name:{}\n{}_acc:{:.2f} {}_auc:{:.2f} {}_precision:{:.2f} {}_sensitivity:{:.2f} {}_specificity:{:.2f} {}_f1score:{:.2f} {}_loc_precision:{:.2f} {}_loc_sensitivity:{:.2f} {}_iou:{:.2f}\n'.format(
                results['model_name'], m, results[f'{m}_acc']*s, m, results[f'{m}_auc']*s, m, results[f'{m}_precision'] *
                    s, m, results[f'{m}_sensitivity']*s, m, results[f'{m}_specificity']*s, m, results[f'{m}_f1score']*s,
                    m, results[f'{m}_loc_precision']*s, m, results[f'{m}_loc_sensitivity']*s,  m, results[f'{m}_iou']*s))

        test_result = []
        for t in threshold:
            print('{} threshold:{}'.format(m, t))
            max_results = None
            for window_size in [1, 2, 3]:
                print('window size:{}'.format(window_size))
                results, y_pred_info = test_model_demo(model, test_loader_demo[0], t, cls_threshold, window_size, np.array(crc_test_path)
                                                       [:, i], demo_model_name, data_path_type, post, device, args.loc)
                plot_predict_info(y_pred_info, m, demo_model_name + '_' + cls_post + '_' + data_path_type)
                with open(demo_results_log_path, 'a+') as f:
                    f.write('{} threshold:{}\nwindow_size={} acc:{:.2f} precision:{:.2f} sensitivity:{:.2f} specificity:{:.2f} f1_score:{:.2f}\n'.format(
                        m, t, window_size, results[0]*s, results[1]*s, results[2]*s, results[3]*s, results[4]*s))
                if max_results is None or max_results[0] < results[0]:
                    max_results = results
            test_result.append(max_results)
            print('\n')
        # plot
        test_result = list(map(list, zip(*test_result)))
        plot_crc_threshold(test_result, threshold, m, demo_model_name + '_' + cls_post + '_' + data_path_type)

    if len(mode_index) == 3:
        test_result = []
        for t in threshold_NAP:
            print('NAP threshold:{}'.format(t))
            max_results = None
            for window_size in [1, 2, 3]:
                print('window size:{}'.format(window_size))
                results, y_pred_info = test_model_demo(model, test_loader_demo[-1], t, window_size, None, demo_model_name, data_path_type, None, device, args.loc)
                plot_predict_info(y_pred_info, 'NAP', demo_model_name + '_' + cls_post + '_' + data_path_type)
                with open(demo_results_log_path, 'a+') as f:
                    f.write('NAP threshold:{}\nwindow_size={} acc:{:.2f} precision:{:.2f} sensitivity:{:.2f} specificity:{:.2f} f1_score:{:.2f}\n'.format(
                        t, window_size, results[0]*s, results[1]*s, results[2]*s, results[3]*s, results[4]*s))
                if max_results is None or max_results[0] < results[0]:
                    max_results = results
            test_result.append(max_results)
            print('\n')
        # plot
        test_result = list(map(list, zip(*test_result)))
        plot_crc_threshold(test_result, threshold_NAP, 'NAP', demo_model_name + '_' + cls_post + '_' + data_path_type)


if __name__ == '__main__':
    args = parse_arguments()
    NUM_MODELS = 1
    INDIVIDUAL_MODEL_RESULTS_FILE = 'run/model_soups/individual_model_results.jsonl'
    UNIFORM_SOUP_RESULTS_FILE = 'run/model_soups/uniform_soup_results.jsonl'
    INDIVIDUAL_GREEDY_SOUP_RESULTS_FILE = 'run/model_soups/individual_greedy_soup_results.jsonl'
    GREEDY_SOUP_RESULTS_FILE = 'run/model_soups/greedy_soup_results.jsonl'

    if not os.path.exists('run/model_soups'):
        os.makedirs('run/model_soups')

    base_model = CRCModel(model_name=args.model_name, loc=args.loc)
    base_model = nn.DataParallel(base_model)

    base_path = os.path.join('run/ckpt', args.ckpt_path)
    model_paths = [os.path.join(base_path, 'best.pth')]

    crc_test_path = []
    data_crc_test = args.data_path

    abnormal_test_path, normal_test_path = [], []
    for line in open(data_crc_test):
        temp_path = ast.literal_eval(line.strip())
        # method 1
        if temp_path[0].find('abnormal') != -1:
            abnormal_test_path.append(temp_path)
        else:
            normal_test_path.append(temp_path)

        # method 2
        crc_test_path.append(ast.literal_eval(line.strip()))

    mode_index = []
    for m in args.mode:
        mode_index.append(MODE_ORDER.index(m))

    # NAP
    if len(mode_index) == 3:
        test_dataset = CRCDataset(crc_test_path, mode='NAP', use_volume=args.use_volume, type='test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=24, pin_memory=True)
        print("Total slices: {}\n".format(len(test_loader.dataset)))
    test_loader_single_mode = []
    for i in mode_index:
        m = MODE_ORDER[i]
        test_dataset_single_mode = CRCDataset(crc_test_path, mode=m, use_volume=args.use_volume, type='test')
        test_loader_single_mode.append(torch.utils.data.DataLoader(test_dataset_single_mode, batch_size=args.batch_size, shuffle=False, num_workers=24, pin_memory=True))
        print("mode {} slices: {}\n".format(m, len(test_loader_single_mode[0].dataset)))

    print('-' * 50 + '\n')

    test_loader_demo = []
    for i in mode_index:
        m = MODE_ORDER[i]
        test_dataset_demo = CRCDemoDataset(crc_test_path, mode=m, use_volume=args.use_volume)
        test_loader_demo.append(torch.utils.data.DataLoader(test_dataset_demo, batch_size=1, shuffle=False, num_workers=24, pin_memory=True))
        print("mode {} patients: {}\n".format(m, len(test_loader_demo[0].dataset)))
    if len(mode_index) == 3:
        test_loader_demo.append(torch.utils.data.DataLoader(
            CRCDemoDataset(crc_test_path, mode='NAP', use_volume=args.use_volume), batch_size=1, shuffle=False, num_workers=24, pin_memory=True))
        print("mode NAP patients: {}\n".format(len(test_loader_demo[-1].dataset)))

    cls_threshold = 0.35
    cls_post = str(cls_threshold).replace('.', '')

    threshold = [t for t in range(3, 9)]
    threshold_NAP = [t for t in range(15, 16)]

    if args.eval_individual_models:
        for j, model_path in enumerate(model_paths):
            assert os.path.exists(model_path)

            state_dict = torch.load(model_path)
            model = base_model
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(device)

            demo_model_name = model_path.split('/')[-2]
            evaluate()

    if args.uniform_soup:
        for j, model_path in enumerate(model_paths):

            print(f'Adding model {j + 1} of {NUM_MODELS} to uniform soup.')

            assert os.path.exists(model_path)
            state_dict = torch.load(model_path)
            if j == 0:
                uniform_soup = {k: v * (1./NUM_MODELS) for k, v in state_dict.items()}
            else:
                uniform_soup = {k: v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

        model = base_model
        model.load_state_dict(uniform_soup)
        model.eval()
        model.to(device)

        temp_model_name = model_paths[0].split('/')[-2]
        demo_model_name = f'uniform_soup_{temp_model_name}'
        evaluate()

        torch.save(model.state_dict(), os.path.join(os.path.dirname(model_paths[0]), 'uniform_soup.pth'))
