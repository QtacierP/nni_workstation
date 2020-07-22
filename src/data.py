from torchvision import datasets, transforms
from utils import make_weights_for_balanced_classes
import torch
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
import os

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