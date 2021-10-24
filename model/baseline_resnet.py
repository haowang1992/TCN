import numpy as np
import numbers

import torch
import torch.nn.functional as F
from torch import nn

from torchvision.models import resnet50

from model.bn import MSSBN1d


class BatchHard(nn.Module):
    def __init__(self, opt):
        super(BatchHard, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(in_features=self.opt.backbone_nhash, out_features=self.opt.wv_size)

    def forward(self, x, label):
        # x size: (Batch_size, Dim)
        dists = self._pairwise_distance(x, self.opt.metric)

        same_identity_mask = torch.eq(label.unsqueeze(dim=1), label.unsqueeze(dim=0))
        positive_mask = torch.logical_xor(same_identity_mask, torch.eye(label.size(0), dtype=torch.bool).to(label.get_device()))

        furthest_positive, _ = torch.max(dists * (positive_mask.float()), dim=1)
        closest_negative, _ = torch.min(dists + 1e8*(same_identity_mask.float()), dim=1)

        diff = furthest_positive - closest_negative
        if isinstance(self.opt.margin, numbers.Real):
            diff = F.relu(diff+self.margin)
        elif self.opt.margin == 'soft':
            diff = F.softplus(diff)
        return diff, self.fc(x)

    def _pairwise_distance(self, x, metric):
        diffs = x.unsqueeze(dim=1) - x.unsqueeze(dim=0)
        if metric == 'sqeuclidean':
            return (diffs **2).sum(dim=-1)
        elif metric == 'euclidean':
            return torch.sqrt(((diffs **2) + 1e-16).sum(dim=-1))
        elif metric == 'cityblock':
            return diffs.abs().sum(dim=-1)


class SemanticBatchHard(nn.Module):
    def __init__(self, opt):
        super(SemanticBatchHard, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(in_features=opt.wv_size, out_features=self.opt.backbone_nhash)

    def forward(self, x, wv, label):
        # x size: (Batch_size, Dim)
        sem_org = self.fc(wv)
        alpha = torch.rand(x.size(0), 1).to(sem_org.get_device())
        sem = alpha * sem_org + (1.0-alpha) * x
        dists = self._pairwise_distance(x, sem, self.opt.metric)

        same_identity_mask = torch.eq(label.unsqueeze(dim=1), label.unsqueeze(dim=0))
        positive_mask = torch.logical_xor(same_identity_mask, torch.eye(label.size(0), dtype=torch.bool).to(label.get_device()))

        furthest_positive, _ = torch.max(dists * (positive_mask.float()), dim=1)
        closest_negative, _ = torch.min(dists + 1e8*(same_identity_mask.float()), dim=1)

        diff = furthest_positive - closest_negative
        if isinstance(self.opt.margin, numbers.Real):
            diff = F.relu(diff+self.margin)
        elif self.opt.margin == 'soft':
            diff = F.softplus(diff)
        return diff, sem

    def _pairwise_distance(self, x, sem, metric):
        diffs = sem.unsqueeze(dim=1) - x.unsqueeze(dim=0)
        if metric == 'sqeuclidean':
            return (diffs **2).sum(dim=-1)
        elif metric == 'euclidean':
            return torch.sqrt(((diffs **2) + 1e-16).sum(dim=-1))
        elif metric == 'cityblock':
            return diffs.abs().sum(dim=-1)


class BaselineConv(nn.Module):
    def __init__(self, opt):
        super(BaselineConv, self).__init__()
        self.opt = opt

        self.pretrained = 'imagenet' if not self.opt.backbone_nopretrained else None
        # part of original
        self.conv1 = resnet50(pretrained=True).conv1
        self.bn1 = resnet50(pretrained=True).bn1
        self.relu = resnet50(pretrained=True).relu
        self.maxpool = resnet50(pretrained=True).maxpool
        self.layer1 = resnet50(pretrained=True).layer1
        self.layer2 = resnet50(pretrained=True).layer2
        self.layer3 = resnet50(pretrained=True).layer3
        self.layer4 = resnet50(pretrained=True).layer4
        self.avgpool = resnet50(pretrained=True).avgpool

    def forward(self, x, y):
        features = self.features(x)
        return features

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class BaselineFC(nn.Module):
    def __init__(self, opt):
        super(BaselineFC, self).__init__()
        self.opt = opt

        if self.opt.dataset_name == 'TUBerlin' or self.opt.dataset_name == 'QuickDraw':
            self.fc_hash = nn.Sequential(
                nn.Linear(in_features=2048, out_features=self.opt.backbone_nhash)
            )
            self.bn = MSSBN1d(num_features=self.opt.backbone_nhash)
            self.drop = nn.Dropout(p=0.5 if np.log2(self.opt.backbone_nhash)<=6 else 0.7)
        else:
            self.fc_hash = nn.Sequential(
                nn.Linear(in_features=2048, out_features=self.opt.backbone_nhash),
                nn.BatchNorm1d(num_features=self.opt.backbone_nhash)
            )
        self.fc_cls = nn.Linear(in_features=self.opt.backbone_nhash, out_features=self.opt.backbone_ncls)
        self.fc_kd = nn.Linear(in_features=self.opt.backbone_nhash, out_features=1000)

        # initialize fc weight
        fc_weight_imagenet = resnet50(pretrained=True, num_classes=1000).fc.weight
        u, s, vh = np.linalg.svd(fc_weight_imagenet.detach().numpy(), full_matrices=True)
        s_new = np.diag(s)[:self.opt.backbone_nhash, :self.opt.backbone_nhash]
        fc_hash_weight = np.dot(s_new, vh[:self.opt.backbone_nhash, :])

        ################### VERY IMPORTANT FOR INITIALIZATION####################
        self.fc_hash[0].weight = nn.Parameter(torch.from_numpy(fc_hash_weight), requires_grad=True)
        fc_kd_weight = u[:, :self.opt.backbone_nhash]
        self.fc_kd.weight = nn.Parameter(torch.from_numpy(fc_kd_weight), requires_grad=True)
        self.fc_kd.bias = resnet50(pretrained=True, num_classes=1000).fc.bias

    def forward(self, sketch_features, image_features):
        sketch_hash, image_hash = self.fc_hash(sketch_features), self.fc_hash(image_features)
        if self.opt.dataset_name == 'TUBerlin' or self.opt.dataset_name == 'QuickDraw':
            self.bn.sketch_flag = True
            sketch_hash = self.drop(self.bn(sketch_hash))
            self.bn.sketch_flag = False
            image_hash = self.drop(self.bn(image_hash))
        sketch_cls, image_cls = self.fc_cls(sketch_hash), self.fc_cls(image_hash)
        image_kd = self.fc_kd(image_hash)
        return sketch_hash, image_hash, sketch_cls, image_cls, image_kd
