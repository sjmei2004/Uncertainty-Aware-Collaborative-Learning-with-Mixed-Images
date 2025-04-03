import argparse
import os
import shutil
from collections import OrderedDict
import pandas as pd
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/UACLMI', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='UNet_feaaddSDM', help='model_name')#UNet_feaaddSDMunet

parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=14,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        # net2.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urpc":
                out_main, _, _, _ = net(input)
            elif FLAGS.model == "UNet_fea":
                out_main, _ = net(input)
            elif FLAGS.model == "UNet_feaaddSDM":
                out_main, _, _ = net(input)
            elif FLAGS.model == "UNet_CCT_CMMT":
                out_main, _,_ = net(input)
            else:
                out_main = net(input)
                # out_main2 = net2(input)
                # out_main = (out_main + out_main2) / 2
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)
    if np.sum(prediction == 2) == 0:
        second_metric = 0, 0, 0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)
    if np.sum(prediction == 3) == 0:
        third_metric = 0, 0, 0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)



    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    metric_dict = OrderedDict()
    metric_dict['name'] = list()
    metric_dict['dice'] = list()
    metric_dict['jaccard'] = list()
    metric_dict['95hd'] = list()
    metric_dict['asd'] = list()
    metric_dict['name'].append(case)
    single_metric = np.mean(np.array([first_metric, second_metric, third_metric]), axis=0)
    metric_dict['dice'].append(single_metric[0])
    metric_dict['jaccard'].append(single_metric[1])
    metric_dict['asd'].append(single_metric[3])
    metric_dict['95hd'].append(single_metric[2])
    metric_csv = pd.DataFrame(metric_dict)
    os.makedirs(test_save_path, exist_ok=True)  # 创建文件夹（如果不存在）
    csv_file_path = os.path.join(test_save_path, 'metric.csv')

    if os.path.exists(csv_file_path):
        # 如果文件存在，读取现有数据并追加新数据
        existing_data = pd.read_csv(csv_file_path)
        updated_data = pd.concat([existing_data, metric_csv], ignore_index=True)
        updated_data.to_csv(csv_file_path, index=False)
    else:
        # 如果文件不存在，直接保存新数据
        metric_csv.to_csv(csv_file_path, index=False)
    # metric_csv.to_csv(test_save_path + '/metric.csv', index=False)

    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    # net = net_factory(net_type='UNet1', in_chns=1,
    #                   class_num=FLAGS.num_classes)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                       class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))

    net.load_state_dict(torch.load(save_mode_path))
   
    print("init weight from {}".format(save_mode_path))
    net.eval()
    # net2.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        print(case, np.mean(np.array([first_metric, second_metric, third_metric]), axis=0))
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)

    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
