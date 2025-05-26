import argparse
from collections import OrderedDict
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from archs.archs import WaveKANet
from torch.optim import lr_scheduler
from tqdm import tqdm
import loss
from datasets import get_dataloaders
from metrics import iou_score
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter
import os

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: )')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--dataseed', default=42, type=int,
                        help='')

    # dataset
    parser.add_argument('--dataset', default='CVC-ClinicDB-Split', help='dataset name')
    parser.add_argument('--data_dir', default='/root/autodl-tmp/data/CVC-ClinicDB-Split', help='dataset dir')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/result', help='output dir')

    # model
    parser.add_argument('--model_name', default='WaveKANet')
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--image_size', default=256, type=int, help='image_size')
    # loss
    parser.add_argument('--loss', default='EnhancedSegmentationLoss')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=300, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'SE': AverageMeter(),
        'PC': AverageMeter(),
        'F1': AverageMeter(),
        'SP': AverageMeter(),
        'ACC': AverageMeter()
    }

    model.train()

    pbar = tqdm(total=len(train_loader))

    for batch in train_loader:
        input = batch['image'].cuda()
        target = batch['mask'].cuda()
        output = model(input)
        loss = criterion(output, target)

        iou, dice, SE, PC, F1, SP, ACC = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        avg_meters['SE'].update(SE, input.size(0))
        avg_meters['PC'].update(PC, input.size(0))
        avg_meters['F1'].update(F1, input.size(0))
        avg_meters['SP'].update(SP, input.size(0))
        avg_meters['ACC'].update(ACC, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('SE', avg_meters['SE'].avg),
            ('PC', avg_meters['PC'].avg),
            ('F1', avg_meters['F1'].avg),
            ('SP', avg_meters['SP'].avg),
            ('ACC', avg_meters['ACC'].avg)
        ])

        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    results = OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('SE', avg_meters['SE'].avg),
        ('PC', avg_meters['PC'].avg),
        ('F1', avg_meters['F1'].avg),
        ('SP', avg_meters['SP'].avg),
        ('ACC', avg_meters['ACC'].avg)
    ])

    return results

def validate(config, val_loader, model, criterion):
    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'SE': AverageMeter(),
        'PC': AverageMeter(),
        'F1': AverageMeter(),
        'SP': AverageMeter(),
        'ACC': AverageMeter()
    }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for batch in val_loader:
            input = batch['image'].cuda()
            target = batch['mask'].cuda()
            output = model(input)
            loss = criterion(output, target)

            iou, dice, SE, PC, F1, SP, ACC = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['SE'].update(SE, input.size(0))
            avg_meters['PC'].update(PC, input.size(0))
            avg_meters['F1'].update(F1, input.size(0))
            avg_meters['SP'].update(SP, input.size(0))
            avg_meters['ACC'].update(ACC, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('SE', avg_meters['SE'].avg),
                ('PC', avg_meters['PC'].avg),
                ('F1', avg_meters['F1'].avg),
                ('SP', avg_meters['SP'].avg),
                ('ACC', avg_meters['ACC'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('SE', avg_meters['SE'].avg),
        ('PC', avg_meters['PC'].avg),
        ('F1', avg_meters['F1'].avg),
        ('SP', avg_meters['SP'].avg),
        ('ACC', avg_meters['ACC'].avg)
    ])

def save_checkpoint(state, save_path):

    torch.save(state, save_path)
    print(f"Checkpoint saved: {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        best_iou = checkpoint['best_iou']
        best_dice = checkpoint['best_dice']
        log = checkpoint['log']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, best_iou, best_dice, log
    return 0, 0, 0, {key: [] for key in ['loss', 'epoch', 'iou', 'dice', 'SE', 'PC', 'F1', 'SP', 'ACC'] +
                     ['val_' + key for key in ['loss', 'iou', 'dice', 'SE', 'PC', 'F1', 'SP', 'ACC']]}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = vars(parse_args())

    if config['name'] is None:
        config['name'] = '%s_%s' % (config['dataset'], config['model_name'])
    os.makedirs('/root/autodl-tmp/result/models/%s' % config['name'], exist_ok=True)
    my_writer = SummaryWriter(f'/root/autodl-tmp/result/models/%s/tf_logs' % config['name'])

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('/root/autodl-tmp/result/models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = getattr(loss, config['loss'])().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['name'])
    model = WaveKANet(num_classes=config['num_classes'], img_size=config['image_size']).to(device)
    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError


    train_loader, val_loader = get_dataloaders(config['data_dir'], config['batch_size'], config['num_workers'], img_size=(config['image_size'], config['image_size']))


    log = {key: [] for key in ['loss', 'epoch', 'iou', 'dice', 'SE', 'PC', 'F1', 'SP', 'ACC']}


    val_keys = ['val_' + key for key in log.keys() if key not in ['epoch']]
    log.update({key: [] for key in val_keys})

    trigger = 0

    checkpoint_path = f'/root/autodl-tmp/result/models/{config["name"]}/checkpoint.pth'


    start_epoch, best_iou, best_dice, log = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])


        log_message = 'loss %.4f - iou %.4f - dice %.4f - SE %.4f - PC %.4f - F1 %.4f - SP %.4f - ACC %.4f' % (
            train_log['loss'], train_log['iou'], train_log['dice'], train_log['SE'],
            train_log['PC'], train_log['F1'], train_log['SP'], train_log['ACC']
        )


        val_log_message = ' - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_SP %.4f - val_ACC %.4f' % (
            val_log['loss'], val_log['iou'], val_log['dice'], val_log['SE'],
            val_log['PC'], val_log['F1'], val_log['SP'], val_log['ACC']
        )

        log_message += val_log_message
        print(log_message)


        log['epoch'].append(epoch)


        for key in train_log.keys():
            if key in log:
                log[key].append(train_log[key])

        for key in val_log.keys():
            val_key = 'val_' + key
            if val_key in log:
                log[val_key].append(val_log[key])


        pd.DataFrame(log).to_csv(f'/root/autodl-tmp/result/models/%s/log.csv' % config['name'], index=False)


        for key in train_log.keys():
            if key in log:
                my_writer.add_scalar(f'train/{key}', train_log[key], global_step=epoch)

        for key in val_log.keys():
            val_key = 'val_' + key
            if val_key in log:
                my_writer.add_scalar(f'val/{key}', val_log[key], global_step=epoch)

        trigger += 1

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'best_iou': best_iou,
            'best_dice': best_dice,
            'log': log
        }

        if (epoch + 1) % 20 == 0:
            save_checkpoint(checkpoint, checkpoint_path)

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), '/root/autodl-tmp/result/models/%s/best_iou_model.pth' % config['name'])
            best_iou = val_log['iou']
            print("=> saved best iou model")
            trigger = 0

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), '/root/autodl-tmp/result/models/%s/best_dice_model.pth' % config['name'])
            best_dice = val_log['dice']
            print("=> saved best dice model")
            trigger = 0


        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            torch.save(model.state_dict(), '/root/autodl-tmp/result/models/%s/last_model.pth' % config['name'])
            print("=> early stopping")
            break
        if epoch == config['epochs'] - 1:
            torch.save(model.state_dict(), '/root/autodl-tmp/result/models/%s/last_model.pth' % config['name'])
            print("=> saved last model")
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()