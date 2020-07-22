import nni
import os
from torchvision import datasets, transforms
import argparse
from utils import make_weights_for_balanced_classes
import torch
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
import logging
import json
from utils import accuracy, quadratic_weighted_kappa
from callback import TorchBar
import numpy as np

# Initialize
args = argparse.ArgumentParser(description='The option of NNI Workstation')
args.add_argument('--config', type=str, default='./kaggle/test.json', help='')
args = args.parse_args()
nni.get_current_parameter()
_logger = logging.getLogger("NNI Workstation")
with open(args.config) as f:
    base_config = json.load(f)
if not os.path.exists(base_config['model_path']):
    os.makedirs(base_config['model_path'])


train_dataloader = None
val_dataloader = None
test_dataloader = None
model = None
loss_func = None
optimizer = None
start_epoch = 0


if 'loss' in base_config['metric']:
    best_metric = np.inf
    opt = np.less
else:
    best_metric = -np.inf
    opt = np.greater


def build(config):
    global train_dataloader
    global val_dataloader
    global test_dataloader
    global model
    global loss_func
    global optimizer
    # ========= Build Data ==============
    if base_config['dataset'] == 'kaggle':
        from data import build_kaggle_dataset
        train_dataloader, val_dataloader, test_dataloader = build_kaggle_dataset(base_config)
    else:
        _logger.error('{} dataset is not supported now'.format(base_config['dataset']))
    # ======== Build Model
    if config['model'] == 'resnet101':
        from torchvision.models import resnet101
        model = resnet101(num_classes=base_config['n_classes'])
    elif config['model'] == 'resnext101':
        from torchvision.models import resnext101_32x8d
        model = resnext101_32x8d(num_classes=base_config['n_classes'])
    elif config['model'] == 'densenet':
        from torchvision.models import densenet121
        model = densenet121(num_classes=base_config['n_classes'])
    else:
        _logger.error('{} model is not supported'.format(config['model']))
    model = torch.nn.DataParallel(model.cuda())
    # Build optimizer
    if base_config['loss'] == 'ce':
        loss_func = torch.nn.CrossEntropyLoss().cuda()
    elif base_config['loss'] == 'bce':
        loss_func = torch.nn.BCELoss().cuda()
    elif base_config['loss'] == 'MSE':
        loss_func = torch.nn.MSELoss().cuda()
    else:
        _logger.error('{} loss is not supported'.format(config['loss']))
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
    if config['optimizer'] == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config['lr'])
    if config['optimizer'] == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config['lr'])
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    if config['optimizer'] == 'Adamax':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])


def train(epoch):
    global train_dataloader
    global val_dataloader
    global test_dataloader
    global model
    global loss_func
    global optimizer
    print('=> Epoch {} <='.format(str(epoch)))
    model.train()
    torch.set_grad_enabled(True)
    n_steps = train_dataloader.__len__()
    bar = TorchBar(target=n_steps, width=30)
    value = []
    for step, batch in enumerate(train_dataloader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        preds = model(x)
        loss = loss_func(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = accuracy(preds, y)
        value.append(('accuracy', acc))
        bar.update(step, value)
    del(bar)

def val(epoch):
    global train_dataloader
    global val_dataloader
    global test_dataloader
    global model
    global loss_func
    global optimizer
    global best_metric
    model.eval()
    c_matrix = np.zeros((base_config['n_classes'], base_config['n_classes']), dtype=int)
    torch.set_grad_enabled(False)
    corrects = 0
    total = 0
    test_loss = 0
    for step, batch in enumerate(val_dataloader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        preds = model(x)
        loss = loss_func(preds, y)
        test_loss += loss.item()
        total += y.size(0)
        acc, correct = accuracy(preds, y, c_matrix)
        corrects += correct
    acc = corrects / total
    kappa = quadratic_weighted_kappa(c_matrix)
    test_loss = test_loss / total
    if base_config['metric'] == 'acc':
        metric = acc
    elif base_config['metric'] == 'kappa':
        metric = kappa
    elif 'loss' in base_config['metric']:
        metric = test_loss
    else:
        logging.error('{} metric is not supported now'.format(base_config['metric']))
    if opt(metric, best_metric):
        model_path = os.path.join(base_config['model_dir'], 'best_pt')
        print('Saving model to {}'.format(model_path))
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'kappa': kappa,
            'loss': test_loss,
            'epoch': epoch,
        }
        torch.save(state, model_path)
        best_metric = metric
    return metric, best_metric


if __name__ == '__main__':
    config = nni.get_next_parameter()
    #config = {"lr": 0.01,"optimizer":"SGD","model":"resnet101"}
    build(config)
    for epoch in range(start_epoch, start_epoch + base_config['epoch']):
        train(epoch)
        metric, best_metric = val(epoch)
        nni.report_intermediate_result(metric)
    nni.report_final_result(best_metric)

