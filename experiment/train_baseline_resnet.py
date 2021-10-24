import argparse
import os
import pickle
import random
import numpy as np
from scipy.spatial.distance import cdist

from sklearn.metrics import average_precision_score
import math
import multiprocessing
from joblib import Parallel, delayed

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, optim

from model.baseline_resnet import BaselineConv, BaselineFC, SemanticBatchHard, BatchHard
from torchvision.models import resnet50
from dataset.data import SketchORImageDataset, SketchImagePairedDataset
from util.tool import adjust_learning_rate, AverageMeter, save_checkpoint, evaluate_metric, compressITQ, \
    RandomSampler, BatchSampler, SoftCrossEntropy


def prepare_environment(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(opt.seed)
    print(f'environment prepared done: {opt}')


def prepare_model(opt):
    sketch_model, image_model = BaselineConv(opt), BaselineConv(opt)
    fc_model = BaselineFC(opt)

    sketch_model = nn.DataParallel(sketch_model).cuda()
    image_model = nn.DataParallel(image_model).cuda()
    fc_model = nn.DataParallel(fc_model).cuda()

    if opt.lambda_sbh > 0.0 and opt.sbh:
        sbh_model = SemanticBatchHard(opt).cuda()
    else:
        sbh_model = BatchHard(opt).cuda()

    model_t = resnet50(pretrained=True, num_classes=1000)
    model_t = nn.DataParallel(model_t).cuda()
    model_t.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_t = SoftCrossEntropy().cuda()

    if opt.lambda_sbh > 0.0 and opt.sbh:
        param_list = [
            {'params': sketch_model.parameters()},
            {'params': image_model.parameters()},
            {'params': fc_model.parameters()},
            {'params': sbh_model.parameters()}
        ]
    else:
        param_list = [
            {'params': sketch_model.parameters()},
            {'params': image_model.parameters()},
            {'params': fc_model.parameters()},
            {'params': sbh_model.parameters()}
        ]

    optimizer = optim.Adam(param_list, lr=opt.lr, weight_decay=opt.wd)
    print(f'model prepared done')
    return sketch_model, image_model, fc_model, sbh_model, model_t, criterion, criterion_t, optimizer


def prepare_dataset(opt):
    dataset_val_sketch = SketchORImageDataset(opt, split='val', aug=False, input_type='sketch')
    dataset_test_sketch = SketchORImageDataset(opt, split='zero', aug=False, input_type='sketch')
    dataset_test_image = SketchORImageDataset(opt, split='zero', aug=False, input_type='image')

    dataset_train_paired = SketchImagePairedDataset(opt, aug=True)

    with open(f'{opt.project_root}/dataset/{opt.dataset_name}/{opt.zero_version}/semantic_train_gwv_dict.pkl', 'rb') as f:
        sem_train = pickle.load(f)
    sem_train = torch.from_numpy(np.array(list(sem_train.values())))
    print(f'dataset prepared done')
    return dataset_val_sketch, dataset_test_sketch, dataset_test_image, dataset_train_paired, sem_train


def prepare_train_dataloader(dataset_train_paired, batch_sampler_paired):
    def worker_init_fn(worker_id):
        np.random.seed(opt.seed + worker_id)
        random.seed(opt.seed + worker_id)

    dataloader_train_paired = DataLoader(dataset=dataset_train_paired, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, batch_sampler=batch_sampler_paired)
    return dataloader_train_paired


def prepare_valtest_dataloader(dataset_val_sketch, dataset_test_sketch, dataset_test_image):
    dataloader_val_sketch = DataLoader(dataset=dataset_val_sketch, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    dataloader_test_sketch = DataLoader(dataset=dataset_test_sketch, batch_size=opt.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    dataloader_test_image = DataLoader(dataset=dataset_test_image, batch_size=opt.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return dataloader_val_sketch, dataloader_test_sketch, dataloader_test_image


def train(opt, sketch_model, image_model, fc_model, sbh_model, model_t, criterion, criterion_t, optimizer, epoch, dataset_train_paired=None):
    losses_cls = AverageMeter()
    losses_sketch_cls, losses_image_cls = AverageMeter(), AverageMeter()
    losses_kd = AverageMeter()
    losses_share = AverageMeter()
    losses_sbh = AverageMeter()

    sketch_model.train()
    image_model.train()
    fc_model.train()
    sbh_model.train()

    random.seed(opt.seed + epoch)
    np.random.seed(opt.seed + epoch)
    sampler_paired = RandomSampler(dataset_train_paired, seed=opt.seed + epoch)
    batch_sampler_paired = BatchSampler(sampler_paired, opt.batch_size, False)
    dataloader_train_paired = prepare_train_dataloader(dataset_train_paired, batch_sampler_paired)

    for i, (sketch_p, image_p, label_p, wv_p) in enumerate(dataloader_train_paired):
        sketch_p, image_p, label_p, wv_p = sketch_p.cuda(), image_p.cuda(), torch.cat([label_p]).cuda(), torch.cat([wv_p]).cuda()
        flag_zero, flag_one = torch.zeros(sketch_p.size(0), 1).cuda(), torch.ones(image_p.size(0), 1).cuda()

        # shuffle for BN to avoid one modality dominate BN
        shuffle_idx = torch.randperm(sketch_p.size(0))
        sketch_p = sketch_p[shuffle_idx]
        image_p = image_p[shuffle_idx]
        label_p = label_p[shuffle_idx]
        wv_p = wv_p[shuffle_idx]

        sketch_feature_all = sketch_model(sketch_p, flag_zero)
        image_feature_all = image_model(image_p, flag_one)
        sketch_hash_all, image_hash_all, sketch_cls_all, image_cls_all, image_kd_all = fc_model(sketch_feature_all, image_feature_all)

        sketch_loss_cls = criterion(sketch_cls_all, label_p)
        image_loss_cls = criterion(image_cls_all, label_p)

        with torch.no_grad():
            image_teacher_all = model_t(image_p)
        loss_kd = criterion_t(image_kd_all, image_teacher_all)

        loss_share = torch.Tensor([0.0]).cuda()
        for ((n1, p1), (n2, p2)) in zip(sketch_model.named_parameters(), image_model.named_parameters()):
            assert n1 == n2
            # do not forget bn in downsample subnetwork
            if ('bn' not in n1) and ('downsample.1' not in n1):
                loss_share += torch.norm(p1 - p2, p='fro')

        if opt.lambda_sbh > 0.0:
            if opt.sbh:
                loss_sbh, x_sem = sbh_model(torch.cat([sketch_hash_all, image_hash_all]), torch.cat([wv_p, wv_p]), torch.cat([label_p, label_p]))
                sft_x_sem = fc_model.module.fc_cls(x_sem)
                entropy = torch.sum(-F.softmax(sft_x_sem, dim=1) * F.log_softmax(sft_x_sem, dim=1), dim=1)
                entropy_weight = (1 + torch.exp(-entropy))
                loss_sbh = (loss_sbh * entropy_weight).mean()
            else:
                entropy = torch.sum(- F.softmax(torch.cat([sketch_cls_all, image_cls_all]), dim=1) * F.log_softmax(torch.cat([sketch_cls_all, image_cls_all]), dim=1), dim=1)
                entropy_weight = (1 + torch.exp(-entropy))
                loss_sbh, x_sem = sbh_model(torch.cat([sketch_hash_all, image_hash_all]), torch.cat([label_p, label_p]))
                loss_sbh = (loss_sbh * entropy_weight).mean() + 1.0*F.mse_loss(x_sem, torch.cat([wv_p, wv_p]))

            losses_sbh.update(opt.lambda_sbh * loss_sbh.item(), sketch_p.size(0))

        loss_cls = sketch_loss_cls + image_loss_cls
        losses_cls.update(opt.lambda_cls * loss_cls.item(), sketch_p.size(0) + image_p.size(0))
        losses_sketch_cls.update(opt.lambda_cls * sketch_loss_cls.item(), sketch_p.size(0))
        losses_image_cls.update(opt.lambda_cls * image_loss_cls.item(), image_p.size(0))
        losses_kd.update(opt.lambda_kd * loss_kd.item(), image_p.size(0))
        losses_share.update(opt.lambda_share * loss_share.item(), 1)


        optimizer.zero_grad()
        loss = opt.lambda_cls * loss_cls + opt.lambda_share * loss_share + opt.lambda_kd * loss_kd
        if opt.lambda_sbh > 0.0:
            loss += opt.lambda_sbh * loss_sbh

        loss.backward()
        optimizer.step()

        if i % opt.print_freq == 0 or i == len(dataloader_train_paired) - 1:
            print(f'Epoch: [{epoch}][{i}/{len(dataloader_train_paired)}], '
                  f'LossSkeCLS {losses_sketch_cls.val:.3f} ({losses_sketch_cls.avg:.3f}), '
                  f'LossImgCLS {losses_image_cls.val:.3f} ({losses_image_cls.avg:.3f}), '
                  f'LossSBH {losses_sbh.val:.3f} ({losses_sbh.avg:.3f}), '
                  f'LossKD {losses_kd.val:.3f} ({losses_kd.avg:.3f}), '
                  f'LossShare {losses_share.val:.3f} ({losses_share.avg:.3f})')


def validate(dataloader, model, fc_model):
    assert dataloader.dataset.type == 'sketch'

    model.eval()
    fc_model.eval()

    pred, gt = [], []
    for i, (sketch, label, wv) in enumerate(dataloader):
        with torch.no_grad():
            feature = model(sketch.cuda(), torch.zeros(sketch.size(0), 1).cuda())
            cls = fc_model(feature, feature)[2]
        pred.extend(torch.max(cls, 1)[1].cpu().numpy().tolist())
        gt.extend(label.tolist())

    pred = np.array(pred, dtype=np.float32)
    gt = np.array(gt, dtype=np.float32)

    unique_label = np.unique(gt)
    acc = 0.0
    for l in unique_label:
        idx = np.nonzero(gt==l)[0]
        acc += np.sum(pred[idx]==gt[idx])/len(idx)
    acc /= unique_label.shape[0]
    print(f'Acc@1: {acc*100.0:.2f}%')
    return acc*100.0


def test(opt, dataloader_sketch, dataloader_image, sketch_model, image_model, fc_model, savedir, validate=False):
    if not validate:
        resume = os.path.join(opt.project_root, 'checkpoint', savedir, 'model_best.pth.tar')

        if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            opt.start_epoch = checkpoint['epoch']

            save_dict = checkpoint['sketch_state_dict']
            model_dict = sketch_model.state_dict()
            trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
            if len(trash_vars) > 0:
                print(f'trashed vars from resume dict: {trash_vars}')
            resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
            model_dict.update(resume_dict)
            sketch_model.load_state_dict(model_dict)

            save_dict = checkpoint['image_state_dict']
            model_dict = image_model.state_dict()
            trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
            if len(trash_vars) > 0:
                print(f'trashed vars from resume dict: {trash_vars}')
            resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
            model_dict.update(resume_dict)
            image_model.load_state_dict(model_dict)

            save_dict = checkpoint['fc_state_dict']
            model_dict = fc_model.state_dict()
            trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
            if len(trash_vars) > 0:
                print(f'trashed vars from resume dict: {trash_vars}')
            resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
            model_dict.update(resume_dict)
            fc_model.load_state_dict(model_dict)
            print("=> loaded checkpoint '{}' (epoch {} acc1 {})".format(resume, checkpoint['epoch'],
                                                                        checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    def get_feature(dataloader, model, fc_model, input_type='sketch'):
        model.eval()
        fc_model.eval()
        feature_list, label_list = [], []
        for i, (data, label, wv) in enumerate(dataloader):
            if i % 10 == 0:
                print(i, end=' ', flush=True)
            with torch.no_grad():
                feature = model(data.cuda(), torch.zeros(data.size(0), 1).cuda() if input_type == 'sketch' else torch.ones(data.size(0), 1).cuda())
                feature = fc_model(feature, feature)[0]
            feature_list.append(F.normalize(feature).cpu().detach().numpy().reshape(data.size(0), -1))
            label_list.append(label.detach().numpy())
        print('')
        feature_all = np.concatenate(feature_list)
        label_all = np.concatenate(label_list)
        print(f'Features ready: {feature_all.shape}, {label_all.shape}')
        return feature_all, label_all

    if opt.eval_protocol == 'iccv19':
        predicted_features_query, gt_labels_query = get_feature(dataloader_sketch, sketch_model, fc_model, input_type='sketch')
        predicted_features_gallery, gt_labels_gallery = get_feature(dataloader_image, image_model, fc_model, input_type='image')
        scores = - cdist(predicted_features_query, predicted_features_gallery)
        np.save(f'{opt.project_root}/checkpoint/{savedir}/score.npy', scores)

        binary_predicted_features_query, binary_predicted_features_gallery = compressITQ(predicted_features_query, predicted_features_gallery)
        binary_scores = - cdist(binary_predicted_features_query, binary_predicted_features_gallery, metric='hamming')
        print('euclidean distance calculated')

        with open(os.path.join(opt.project_root, 'checkpoint', savedir, 'features_zero.pickle'), 'wb') as fh:
            pickle.dump([predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, None], fh)
        print('feature stored')

        smap, sprec = evaluate_metric(predicted_features_query, gt_labels_query, gt_labels_gallery, scores, top=opt.topk if opt.topk > 0 else None)
        if not validate:
            _, _ = evaluate_metric(binary_predicted_features_query, gt_labels_query, gt_labels_gallery, binary_scores, top=opt.topk if opt.topk > 0 else None)
        return smap, sprec
    else:
        predicted_features_query, gt_labels_query = get_feature(dataloader_sketch, sketch_model, fc_model, input_type='sketch')
        predicted_features_gallery, gt_labels_gallery = get_feature(dataloader_image, image_model, fc_model, input_type='image')
        distance = cdist(predicted_features_query, predicted_features_gallery, 'euclidean')
        sim = 1 / (1 + distance)
        str_sim = (np.expand_dims(gt_labels_query, axis=1) == np.expand_dims(gt_labels_gallery, axis=0)) * 1

        nq = str_sim.shape[0]
        num_cores = min(multiprocessing.cpu_count(), 32)

        # -sim because values in similarity means 0= un-similar 1= very-similar
        arg_sort_sim = (-sim).argsort()
        sort_sim = []
        sort_lst = []
        for indx in range(0, arg_sort_sim.shape[0]):
            sort_sim.append(sim[indx, arg_sort_sim[indx, :]])
            sort_lst.append(str_sim[indx, arg_sort_sim[indx, :]])

        sort_sim = np.array(sort_sim)
        sort_str_sim = np.array(sort_lst)

        aps_200 = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(sort_str_sim[iq, 0:200], sort_sim[iq, 0:200]) for iq in range(nq))
        aps_200_actual = [0.0 if math.isnan(x) else x for x in aps_200]
        map_200 = np.mean(aps_200_actual)

        # Precision@200 means at the place 200th
        precision_200 = np.mean(sort_str_sim[:, 200])

        # aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))
        # map_ = np.mean(aps)
        return map_200, precision_200


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch experiment for training zero-shot sketch-based image retrieval')
    # project specific
    parser.add_argument('--project_root', type=str, default='/home/wanghao/RemoteProject/TCN/')
    # dataset specific
    parser.add_argument('--dataset_root', type=str, default='/home/wanghao/Datasets/ZS-SBIR/SAKE/')
    parser.add_argument('--dataset_name', type=str, default='Sketchy', choices=['Sketchy', 'TUBerlin', 'QuickDraw'])
    parser.add_argument('--zero_version', type=str, default='zeroth')
    # model specific
    parser.add_argument('--arch', type=str, default='baseline_resnet')
    parser.add_argument('--backbone_nopretrained', action='store_true', default=False)
    parser.add_argument('--backbone_nhash', type=int, default=512)
    parser.add_argument('--backbone_ncls', type=int, default=220)
    parser.add_argument('--wv_size', type=int, default=300)
    # training specific
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nepoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--testing', action='store_true', default=False)
    # parameters
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_kd', type=float, default=1.0)
    parser.add_argument('--lambda_share', type=float, default=1000.0)
    parser.add_argument('--metric', type=str, default='euclidean')
    parser.add_argument('--margin', default='soft')
    parser.add_argument('--sbh', action='store_true', default=False)
    parser.add_argument('--lambda_sbh', type=float, default=1.0)
    parser.add_argument('--sbh_ablation', action='store_true', default=False)
    parser.add_argument('--eval_protocol', type=str, default='iccv19', choices=['iccv19', 'cvpr19'])
    parser.add_argument('--topk', type=int, default=0, choices=[0, 200])
    parser.add_argument('--sp', action='store_true', default=False)
    parser.add_argument('--lambda_sp', type=float, default=1.0)
    opt = parser.parse_args()

    prepare_environment(opt)
    sketch_model, image_model, fc_model, sbh_model, model_t, criterion, criterion_t, optimizer = prepare_model(opt)
    dataset_val_sketch, dataset_test_sketch, dataset_test_image, dataset_train_paired, sem_train = prepare_dataset(opt)
    dataloader_val_sketch, dataloader_test_sketch, dataloader_test_image = prepare_valtest_dataloader(dataset_val_sketch, dataset_test_sketch, dataset_test_image)

    best_acc1 = 0.0
    savedir = f'' \
        f'Dataset({opt.dataset_name})_Arch({opt.arch})_' \
        f'BackboneNoPretrained({opt.backbone_nopretrained})_' \
        f'LR({opt.lr})_BS({opt.batch_size})_WD({opt.wd})_' \
        f'Lambda_Cls({opt.lambda_cls})_KD({opt.lambda_kd})_Share({opt.lambda_share})_SBH({opt.sbh})_({opt.lambda_sbh})_' \
        f'Seed({opt.seed})_' \
        f'Dim({opt.backbone_nhash})'

    if not os.path.exists(f'{opt.project_root}/checkpoint/{savedir}'):
        os.makedirs(f'{opt.project_root}/checkpoint/{savedir}')

    if opt.testing:
        test(opt, dataloader_test_sketch, dataloader_test_image, sketch_model, image_model, fc_model, savedir)
    else:
        for epoch in range(opt.nepoch):
            adjust_learning_rate(opt, optimizer, epoch)

            train(opt, sketch_model, image_model, fc_model, sbh_model, model_t, criterion, criterion_t, optimizer, epoch, dataset_train_paired=dataset_train_paired)

            acc1 = validate(dataloader_val_sketch, sketch_model, fc_model)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_checkpoint({
                'epoch': epoch + 1,
                'sketch_state_dict': sketch_model.state_dict(),
                'image_state_dict': image_model.state_dict(),
                'fc_state_dict': fc_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=f'{opt.project_root}/checkpoint/{savedir}/checkpoint.pth.tar')
