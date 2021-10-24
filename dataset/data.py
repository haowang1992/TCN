import os
import cv2
import pickle
import numpy as np

from skimage.transform import warp, AffineTransform
from torch.utils.data import Dataset

import torchvision.transforms as T


immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
imstd = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.ToPILImage(),
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(immean, imstd)
])


def random_parameter(threshold=0.5):
    if np.random.random() < threshold:
        hflip = True
    else:
        hflip = False

    if np.random.random() < threshold:
        sx = np.random.uniform(0.7, 1.3)
        sy = np.random.uniform(0.7, 1.3)
    else:
        sx = 1.0
        sy = 1.0

    if np.random.random() < threshold:
        rx = np.random.uniform(-30.0 * 2.0 * np.pi / 360.0, +30.0 * 2.0 * np.pi / 360.0)
    else:
        rx = 0.0

    if np.random.random() < threshold:
        tx = np.random.uniform(-10, 10)
        ty = np.random.uniform(-10, 10)
    else:
        tx = 0.0
        ty = 0.0
    return hflip, sx, sy, rx, tx, ty


def random_transform(img, hflip, sx, sy, rx, tx, ty):
    if hflip:
        img = img[:, ::-1, :]
    aftrans = AffineTransform(scale=(sx, sy), rotation=rx, translation=(tx, ty))
    img_aug = warp(img, aftrans.inverse, preserve_range=True).astype('uint8')
    return img_aug


class SketchImagePairedDataset(Dataset):
    def __init__(self, opt, aug=False, shuffle=False, first_n_debug=9999999):
        self.opt = opt
        self.root_dir = f'{self.opt.project_root}/dataset/{self.opt.dataset_name}'
        self.dataset_dir = f'{self.opt.dataset_root}/{self.opt.dataset_name}'
        if self.opt.dataset_name == 'Sketchy' or self.opt.dataset_name == 'QuickDraw':
            self.sketch_version = 'sketch_tx_000000000000_ready'
            self.image_version = 'all_photo'
        elif self.opt.dataset_name == 'TUBerlin':
            self.sketch_version = 'png_ready'
            self.image_version = 'ImageResized_ready'
        self.zero_version = self.opt.zero_version
        self.transform = transform
        self.aug = aug

        file_ls_sketch = os.path.join(self.root_dir, self.zero_version, self.sketch_version + '_filelist_train.txt')
        file_ls_image = os.path.join(self.root_dir, self.zero_version, self.image_version + '_filelist_train.txt')
        file_dict_semantic = os.path.join(self.root_dir, self.zero_version, 'semantic_gwv_dict.pkl')

        with open(file_ls_sketch, 'r') as fh:
            file_content_sketch = fh.readlines()
        with open(file_ls_image, 'r') as fh:
            file_content_image = fh.readlines()
        with open(file_dict_semantic, 'rb') as fh:
            self.semantic_wv_dict = pickle.load(fh)

        self.file_ls_sketch = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content_sketch])
        self.labels_sketch = np.array([int(ff.strip().split()[-1]) for ff in file_content_sketch])
        self.names_sketch = np.array([' '.join(ff.strip().split()[:-1]).split('/')[-2] for ff in file_content_sketch])
        self.file_ls_image = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content_image])
        self.labels_image = np.array([int(ff.strip().split()[-1]) for ff in file_content_image])
        self.names_image = np.array([' '.join(ff.strip().split()[:-1]).split('/')[-2] for ff in file_content_image])

        if shuffle:
            self.shuffle()

        self.file_ls_sketch = self.file_ls_sketch[:first_n_debug]
        self.labels_sketch = self.labels_sketch[:first_n_debug]
        self.names_sketch = self.names_sketch[:first_n_debug]
        self.file_ls_image = self.file_ls_image[:first_n_debug]
        self.labels_image = self.labels_image[:first_n_debug]
        self.names_image = self.names_image[:first_n_debug]

    def __getitem__(self, idx):
        label = self.labels_image[idx]
        wv = self.semantic_wv_dict[self.names_image[idx]]
        hflip, sx, sy, rx, tx, ty = random_parameter()

        select_idx = np.random.choice(np.argwhere(self.labels_sketch == label).reshape(-1), 1)
        sketch = cv2.imread(os.path.join(self.dataset_dir, self.file_ls_sketch[select_idx][0]))[:, :, ::-1]
        if self.aug:
            sketch = random_transform(sketch, hflip, sx, sy, rx, tx, ty)
        if self.transform is not None:
            sketch = self.transform(sketch)

        image = cv2.imread(os.path.join(self.dataset_dir, self.file_ls_image[idx]))[:, :, ::-1]
        if self.aug:
            image = random_transform(image, hflip, sx, sy, rx, tx, ty)
        if self.transform is not None:
            image = self.transform(image)
        return sketch, image, label, wv

    def __len__(self):
        return len(self.labels_image)

    def shuffle(self):
        s_idx = np.arange(len(self.labels_sketch))
        np.random.shuffle(s_idx)
        self.file_ls_sketch = self.file_ls_sketch[s_idx]
        self.labels_sketch = self.labels_sketch[s_idx]
        self.names_sketch = self.names_sketch[s_idx]
        s_idx = np.arange(len(self.labels_image))
        np.random.shuffle(s_idx)
        self.file_ls_image = self.file_ls_image[s_idx]
        self.labels_image = self.labels_image[s_idx]
        self.names_image = self.names_image[s_idx]


class SketchORImageDataset(Dataset):
    def __init__(self, opt, split='train', aug=False, shuffle=False, first_n_debug=9999999, input_type='sketch'):
        self.opt = opt
        self.root_dir = f'{self.opt.project_root}/dataset/{self.opt.dataset_name}'
        self.dataset_dir = f'{self.opt.dataset_root}/{self.opt.dataset_name}'
        if self.opt.dataset_name == 'Sketchy' or self.opt.dataset_name == 'QuickDraw':
            self.sketch_version = 'sketch_tx_000000000000_ready'
            self.image_version = 'all_photo'
        elif self.opt.dataset_name == 'TUBerlin':
            self.sketch_version = 'png_ready'
            self.image_version = 'ImageResized_ready'
        self.zero_version = self.opt.zero_version
        self.transform = transform
        self.split = split
        self.aug = aug
        self.type = input_type

        if self.split == 'train':
            if self.type == 'sketch':
                file_ls = os.path.join(self.root_dir, self.zero_version, self.sketch_version + '_filelist_train.txt')
            elif self.type == 'image':
                file_ls = os.path.join(self.root_dir, self.zero_version, self.image_version + '_filelist_train.txt')
        elif self.split == 'val':
            assert self.type == 'sketch'
            file_ls = os.path.join(self.root_dir, self.zero_version, self.sketch_version + '_filelist_test.txt')
        elif self.split == 'zero':
            if self.type == 'sketch':
                file_ls = os.path.join(self.root_dir, self.zero_version, self.sketch_version + '_filelist_zero.txt')
                # file_ls = os.path.join(self.root_dir, self.zero_version, self.sketch_version + '_filelist_train.txt')
            elif self.type == 'image':
                file_ls = os.path.join(self.root_dir, self.zero_version, self.image_version + '_filelist_zero.txt')
                # file_ls = os.path.join(self.root_dir, self.zero_version, self.image_version + '_filelist_train.txt')

        else:
            print('unknown split for dataset initialization: ' + self.split)
            return
        file_dict_semantic = os.path.join(self.root_dir, self.zero_version, 'semantic_gwv_dict.pkl')

        with open(file_ls, 'r') as fh:
            file_content = fh.readlines()
        with open(file_dict_semantic, 'rb') as fh:
            self.semantic_wv_dict = pickle.load(fh)

        self.files = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        self.names = np.array([' '.join(ff.strip().split()[:-1]).split('/')[-2] for ff in file_content])

        if shuffle:
            assert self.split == 'train'
            self.shuffle()

        self.files = self.files[:first_n_debug]
        self.labels = self.labels[:first_n_debug]
        self.names = self.names[:first_n_debug]

    def __getitem__(self, idx):
        label = self.labels[idx]
        wv = self.semantic_wv_dict[self.names[idx]]
        data = cv2.imread(os.path.join(self.dataset_dir, self.files[idx]))[:, :, ::-1]
        if self.aug:
            hflip, sx, sy, rx, tx, ty = random_parameter()
            data = random_transform(data, hflip, sx, sy, rx, tx, ty)
        if self.transform is not None:
            data = self.transform(data)
        return data, label, wv

    def __len__(self):
        return len(self.labels)

    def shuffle(self):
        s_idx = np.arange(len(self.labels))
        np.random.shuffle(s_idx)
        self.files = self.files[s_idx]
        self.labels = self.labels[s_idx]
        self.names = self.names[s_idx]
