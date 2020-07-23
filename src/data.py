from torchvision import datasets, transforms
from utils import make_weights_for_balanced_classes
import torch
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
import os
import PIL.Image as Image
import numpy as np
import random
import glob
from PIL.Image import NEAREST

def build_kaggle_dataset(base_config):
    data_dir = '/data2/chenpj/FundusTool/data/fundus/EyePac/'
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    val_path = os.path.join(data_dir, 'val')
    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(
            size=base_config['size'],
            scale=(1 / 1.15, 1.15),
            ratio=(0.7561, 1.3225)
        ),
        transforms.RandomAffine(
            degrees=(-180, 180),
            translate=(40 / base_config['size'], 40 / base_config['size']),
            scale=None,
            shear=None
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(base_config['mean'],
                             base_config['std']),
        # KrizhevskyColorAugmentation(sigma=0.5)
    ])
    test_preprocess = transforms.Compose([
        transforms.Resize((base_config['size'], base_config['size'])),
        transforms.ToTensor(),
        transforms.Normalize(base_config['mean'],
                             base_config['std'])
    ])
    # Compile Dataset
    train_dataset = datasets.ImageFolder(train_path, train_preprocess)
    test_dataset = datasets.ImageFolder(test_path, test_preprocess)
    val_dataset = datasets.ImageFolder(val_path, test_preprocess)
    weights, weights_per_class = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
    print('Use sample weights')
    # weights= torch.DoubleTensor(weights)
    # Compile Sampler
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=False)
    print('[Train]: ', train_dataset.__len__())
    print('[Val]: ', val_dataset.__len__())
    print('[Test]: ', test_dataset.__len__())
    train_dataloader = DataLoader(train_dataset, batch_size=base_config['batch_size'],
                                  sampler=weighted_sampler, num_workers=base_config['n_threads'])
    val_dataloader = DataLoader(val_dataset, batch_size=base_config['batch_size'],
                                shuffle=True, num_workers=base_config['n_threads'])
    test_dataloader = DataLoader(test_dataset, batch_size=base_config['batch_size'],
                                 shuffle=False, num_workers=base_config['n_threads'])
    return train_dataloader, val_dataloader, test_dataloader


class DRIVEDataset(torch.utils.data.Dataset):
    def __init__(self,root, transforms, gt_transforms, step=1):
        super(DRIVEDataset, self).__init__()
        self.root = root
        self.step = step
        self.transforms = transforms
        self.gt_transforms = gt_transforms
        self.imgs_path =  os.path.join(root, 'origin')
        self.labels_path = os.path.join(root, 'groundtruth')
        self.imgs_list =  sorted(glob.glob(os.path.join(self.imgs_path, '*')))
        self.labels_list =  sorted(glob.glob(os.path.join(self.labels_path, '*')))

    def __len__(self):
        return len(self.imgs_list) * self.step

    def __getitem__(self, index):
        img = Image.open(self.imgs_list[index % len(self.imgs_list)])
        label = Image.open(self.labels_list[index % len(self.imgs_list)])
        # Keep the same transformation
        seed = np.random.randint(999999999)
        random.seed(seed)
        label = self.gt_transforms(label)
        random.seed(seed)
        img = self.transforms(img)
        return img, label


def build_drive_dataset(base_config):
    train_preprocess = transforms.Compose([
        transforms.Resize((base_config['size'], base_config['size']), interpolation=NEAREST),
        transforms.RandomCrop((base_config['crop_size'], base_config['crop_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(base_config['mean'],
                             base_config['std']),
    ])
    train_gt_preprocess = transforms.Compose([
        transforms.Resize((base_config['size'], base_config['size']), interpolation=NEAREST),
        transforms.RandomCrop((base_config['crop_size'], base_config['crop_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),

    ])

    test_preprocess = transforms.Compose([
        transforms.Resize((base_config['size'], base_config['size']), interpolation=NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(base_config['mean'],
                             base_config['std']),
    ])

    test_gt_preprocess = transforms.Compose([
        transforms.Resize((base_config['size'], base_config['size']), interpolation=NEAREST),
        transforms.ToTensor(),
    ])
    n = base_config['size'] // base_config['crop_size']
    step = n * n
    train_dataset = DRIVEDataset('/data2/chenpj/FundusTool/data/vessel/DRIVE/train',
                                 train_preprocess, train_gt_preprocess, step=step)
    val_dataset = DRIVEDataset('/data2/chenpj/FundusTool/data/vessel/DRIVE/validate',
                               test_preprocess, test_gt_preprocess)
    test_dataset = DRIVEDataset('/data2/chenpj/FundusTool/data/vessel/DRIVE/test',
                           test_preprocess, test_gt_preprocess)
    print('[Train]: ', train_dataset.__len__() * step)
    print('[Val]: ', val_dataset.__len__())
    print('[Test]: ', test_dataset.__len__())
    train_dataloader = DataLoader(train_dataset, batch_size=base_config['batch_size'],
                                  shuffle=True, num_workers=base_config['n_threads'])
    val_dataloader = DataLoader(val_dataset, batch_size=base_config['test_batch_size'],
                                shuffle=False, num_workers=base_config['n_threads'])
    test_dataloader = DataLoader(test_dataset, batch_size=base_config['test_batch_size'],
                                 shuffle=False, num_workers=base_config['n_threads'])
    return train_dataloader, val_dataloader, test_dataloader