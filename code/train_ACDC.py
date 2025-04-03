import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from skimage.measure import label
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.losses import loss_calc
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volumecisr

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/UACLMI', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='UNet_feaaddSDM', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=6,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--ohem', type=bool, default=False)
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
args = parser.parse_args()


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

dice_loss = losses.BCPDiceLoss(n_classes=4)
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def Weighted_GAP(supp_feat, mask):
    if len(mask.size()) != 4:  # 12,256,256
        mask = mask.unsqueeze(1).float()  # 12,1,256,256
    if supp_feat.size() != mask.size():  # 12,32,128,128  12,1,256,256
        supp_feat = F.interpolate(supp_feat, size=mask.size()[-2:], mode='bilinear')  # 12,32,256,256

    supp_feat = supp_feat * mask  # 12,32,256,256
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005  # 4,1,1,1
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def consist(lafeat1, map1, mask, unfeat1, pred1, unpred_pseudo2):
    weight = []
    loss_mat = 0
    loss_simi = 0.0
    for i in range(4):
        map = map1  # 12,4,256,256
        a = map[:, i, :, :]  # 12,256,256
        vec1 = Weighted_GAP(lafeat1, map[:, i, :, :]).mean(dim=0, keepdim=True)  # 12,32,1,1   fea12,32,128,128

        simi_map1 = F.cosine_similarity(unfeat1, vec1, dim=1)  # 12,128,128

        simi_map1 = F.interpolate(simi_map1.unsqueeze(1), size=map1.size()[-2:], mode='bilinear')#.detach()  # 12,1,256,256

        high_simi_map1 = simi_map1.add(1).mul(0.5)
        class_mask = high_simi_map1 > 0.8
        class_mask = class_mask.float()
        high_vec = Weighted_GAP(unfeat1,class_mask).mean(dim=0, keepdim=True)#1,32,1,1
        protype = 0.9 * vec1 + 0.1 * high_vec
        simi_map1 = F.cosine_similarity(unfeat1, protype, dim=1)  # 12,128,128
        simi_map1 = F.interpolate(simi_map1.unsqueeze(1), size=map1.size()[-2:], mode='bilinear')  # 12,1,256,256

        simi_map1 = simi_map1.view(map[:, i, :, :].size())  # 12,256,256


        simi_mapla = F.cosine_similarity(lafeat1, protype, dim=1)  # 12,128,128
        simi_mapla = F.interpolate(simi_mapla.unsqueeze(1), size=map1.size()[-2:], mode='bilinear')  # 12,1,256,256
        simi_mapla = simi_mapla.view(map[:, i, :, :].size())  # 12,256,256

        loss_simi += torch.mean((simi_mapla - a) ** 2)
        simi_map1 = simi_map1.unsqueeze(dim=1)

        weight.append(simi_map1)
    weights = torch.cat(weight, dim=1)  # 12,4,256,256

    weights = torch.softmax(weights,dim=1)


    max_indices = torch.argmax(pred1, dim=1)  # 12,256,256

    loss_mat += loss_calc(args, pred1, unpred_pseudo2, reduction='none')
    max_indices = torch.argmax(pred1, dim=1)
    gathered_weights = torch.gather(weights, 1, unpred_pseudo2.unsqueeze(1))
    loss_mat = (gathered_weights.unsqueeze(1) * loss_mat).mean()

    return loss_mat, loss_simi * 0.125, gathered_weights

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def getbox_new(size, dxy):
    W = size[2]
    H = size[3]

    cx = np.random.randint(0, W, size[0])
    cy = np.random.randint(0, H, size[0])

    bbx1 = np.clip(cx - dxy[0] // 2, 0, W)
    bbx2 = np.clip(bbx1 + dxy[0], 0, W)
    bbx1 = np.clip(bbx2 - dxy[0], 0, W)
    bby1 = np.clip(cy - dxy[1] // 2, 0, H)
    bby2 = np.clip(bby1 + dxy[1], 0, H)
    bby1 = np.clip(bby2 - dxy[1], 0, H)

    return bbx1, bby1, bbx2, bby2

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs
def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()
    ema_model = create_model(ema=True)
    model1.train()
    model2.train()
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)


    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    model_loss = losses.mse_loss
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[
                                                                                               args.labeled_bs + unlabeled_sub_bs:]
            # ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[
            #                                                                                   args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                pre_a,_,_ = ema_model(uimg_a)
                pre_b,_,_ = ema_model(uimg_b)

                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)

            rate = 1 / 4
            uimg_a_small = F.interpolate(uimg_a, scale_factor=rate)
            uimg_a_plab_small = F.interpolate(plab_a.unsqueeze(0), scale_factor=rate)
            uimg_a_plab_small = uimg_a_plab_small.squeeze(0)
            img_b_small = F.interpolate(img_b, scale_factor=rate)
            img_b_lab_small = F.interpolate(lab_b.unsqueeze(0), scale_factor=rate)
            img_b_lab_small = img_b_lab_small.squeeze(0)
            m1, n1, m2, n2 = getbox_new(uimg_a.size(), uimg_a_small.shape[2:])
            rand_index = torch.randperm(uimg_a.size()[0])
            net_input_unl = img_a.clone()
            net_input_unl_lab = lab_a.clone()
            net_input_l = uimg_b.clone()
            net_input_l_lab = plab_b.clone()
            for i in range(img_a.shape[0]):
                net_input_unl[i, :, m1[i]:m2[i], n1[i]:n2[i]] = uimg_a_small[rand_index[i]]
                net_input_unl_lab[i, m1[i]:m2[i], n1[i]:n2[i]] = uimg_a_plab_small[rand_index[i]]
                net_input_l[i, :, m1[i]:m2[i], n1[i]:n2[i]] = img_b_small[rand_index[i]]
                net_input_l_lab[i, m1[i]:m2[i], n1[i]:n2[i]] = img_b_lab_small[rand_index[i]]


            out_unl,_,_ = model1(net_input_unl)
            out_l,_,_ = model1(net_input_l)




            loss_dice = dice_loss(torch.softmax(out_unl, dim=1), net_input_unl_lab.unsqueeze(1)) + dice_loss(torch.softmax(out_l, dim=1), net_input_l_lab.unsqueeze(1))

            mixloss = loss_dice
            tensor_list = []
            tensor = label_batch.unsqueeze(dim=1)
            for i in range(4):
                temp_prob = tensor[:args.labeled_bs, :, :, :] == i * torch.ones_like(tensor[:args.labeled_bs, :, :, :])
                tensor_list.append(temp_prob)
            output_tensor = torch.cat(tensor_list, dim=1)  # 2,4,256,256
            output_tensor = output_tensor.float()
            map = output_tensor

            outputs1, fea1, tanh1 = model1(volume_batch)  # 24,32,128,128
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs2, fea2, tanh2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            lafeat1 = fea1[:args.labeled_bs, :, :, :]
            lafeat2 = fea2[:args.labeled_bs, :, :, :]
            mask = label_batch[:args.labeled_bs]

            unfeat1 = fea1[args.labeled_bs:, :, :, :]
            unfeat2 = fea2[args.labeled_bs:, :, :, :]

            consistency_weight = get_current_consistency_weight(iter_num // 150)


            loss_sdm1 = 0
            loss_sdm2 = 0
            loss_sdm1 += model_loss(tanh1[args.labeled_bs:], tanh2[args.labeled_bs:].detach())
            loss_sdm2 += model_loss(tanh2[args.labeled_bs:], tanh1[args.labeled_bs:].detach())

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
            unpred_pseudo2 = pseudo_outputs2
            pred1 = outputs1[args.labeled_bs:]
            unpred_pseudo1 = pseudo_outputs1
            pred2 = outputs2[args.labeled_bs:]
            pseudo_supervision1, loss_simi1, uncertainty1 = consist(lafeat2, map, mask, unfeat2, pred1, unpred_pseudo2)
            pseudo_supervision2, loss_simi2, uncertainty2 = consist(lafeat1, map, mask, unfeat1, pred2, unpred_pseudo1)


            if iter_num>=1000:
                model1_loss = loss1 + 0.3 * pseudo_supervision1  + 0.3 * loss_sdm1 + 0.3 * mixloss+ loss_simi1
                model2_loss = loss2 + 0.3 * pseudo_supervision2  + 0.3 * loss_sdm2+ loss_simi2
            else:
                model1_loss = loss1 + 0.3 * pseudo_supervision1 + 0.3 * mixloss+ loss_simi1
                model2_loss = loss2 + 0.3 * pseudo_supervision2 + loss_simi2


            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()
            update_ema_variables(model1, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num % 100 == 0:
                image = img_a[0, 0:1, :, :]
                writer.add_image('train/img_a', image, iter_num)

                image = uimg_a[0, 0:1, :, :]
                writer.add_image('train/uimg_a', image, iter_num)

                image = net_input_unl[0, 0:1, :, :]
                writer.add_image('train/Imageunl', image, iter_num)

                image_lab = net_input_unl_lab[0, ...].unsqueeze(0) * 50
                writer.add_image('train/Imageunl_lab', image_lab, iter_num)

                outputs = torch.argmax(torch.softmax(
                    out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Imageunl_output',
                                 outputs[0, ...] * 50, iter_num)
                max_values, max_indices = torch.max(tanh1, dim=1)
                # outputs = torch.argmax(torch.softmax(
                #     tanh1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/tanhSDM',
                                 max_values[0].unsqueeze(0) * 50, iter_num)

                uncertainty_image = uncertainty1[0,0:1,:, :] * 50

                fig, ax = plt.subplots(figsize=(6, 6))
                # 转换为 numpy 数组以便使用 matplotlib
                uncertainty_np = uncertainty_image[0].detach().cpu().numpy()
                # 使用热图 (cmap='hot' 产生从红到黄的渐变颜色)
                cax = ax.imshow(uncertainty_np, cmap='hot', interpolation='nearest')
                # 关闭坐标轴
                ax.axis('off')
                # 保存图像到内存
                plt.colorbar(cax)  # 添加 colorbar 显示色条
                plt.tight_layout()
                plt.close(fig)
                # 转换为图片格式
                import io
                from PIL import Image
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                img_array = np.array(img)
                # 将图像传递给 TensorBoard
                writer.add_image('train/uncertainty', img_array.transpose(2, 0, 1), iter_num)
                buf.close()

                image = volume_batch[0, 0:1, :, :]
                writer.add_image('train/volume_batchl', image, iter_num)
                image = volume_batch[6, 0:1, :, :]
                writer.add_image('train/volume_batchunl', image, iter_num)

                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[6, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[6, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volumecisr(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()
                ema_model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volumecisr(
                        sampled_batch["image"], sampled_batch["label"], ema_model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/ema_model_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/ema_model_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/ema_model_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/ema_model_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'ema_model_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_ema_model.pth'.format(args.model))
                    torch.save(ema_model.state_dict(), save_mode_path)
                    torch.save(ema_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : ema_model_mean_dice : %f ema_model_mean_hd95 : %f' % (
                    iter_num, performance2, mean_hd952))
                ema_model.train()
                

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))
                save_mode_path = os.path.join(
                    snapshot_path, 'ema_model_iter_' + str(iter_num) + '.pth')
                torch.save(ema_model.state_dict(), save_mode_path)
                logging.info("save ema_model to {}".format(save_mode_path))
                # save_mode_path = os.path.join(
                #     snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                # torch.save(model2.state_dict(), save_mode_path)
                # logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:                                                                                                                                                                                                                                                                                                   
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False                                                                                                                                                                                                                                                                                                                   
        cudnn.deterministic = True

    random.seed(args.seed)                                                                                                                                                                                                                                                                                                                                                                                                                      
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)                                                                                                                                                                                                               
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)                                                                                               
    if not os.path.exists(snapshot_path):                                                                                                                                                                
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
