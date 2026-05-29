import os
import sys
import time
import glob
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset

# Ensure cnn/ is importable
_ROOT = os.path.dirname(__file__)
_CNN = os.path.join(_ROOT, 'cnn')
for _p in [_ROOT, _CNN]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import utils                              # cnn/utils.py
import genotypes as _genotypes_module     # cnn/genotypes.py
from genotypes import Genotype            # needed for eval() in _load_genotype
from model import NetworkCIFAR as Network  # cnn/model.py


parser = argparse.ArgumentParser('Network Train')
parser.add_argument('--data', default='./data')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--learning_rate', type=float, default=0.025)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=3e-4)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--report_freq', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
# Architecture
parser.add_argument('--arch', default=None,
                    help='named genotype from cnn/genotypes.py (e.g. DARTS_V2)')
parser.add_argument('--genotype_file', default=None,
                    help='path to best_genotype.txt produced by dynonas_search.py')
parser.add_argument('--init_channels', type=int, default=36)
parser.add_argument('--layers', type=int, default=20,
                    help='total cells')
parser.add_argument('--auxiliary', action='store_true', default=False)
parser.add_argument('--auxiliary_weight', type=float, default=0.4)
parser.add_argument('--cutout', action='store_true', default=False)
parser.add_argument('--cutout_length', type=int, default=16)
parser.add_argument('--drop_path_prob', type=float, default=0.3)
parser.add_argument('--save', default='eval-dynonas')
args = parser.parse_args()

args.save = '{}-{}'.format(args.save, time.strftime('%Y%m%d-%H%M%S'))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger().addHandler(fh)


def _load_genotype():
    if args.genotype_file:
        with open(args.genotype_file) as f:
            return eval(f.read().strip())
    if args.arch:
        return getattr(_genotypes_module, args.arch)
    raise ValueError('Provide --arch or --genotype_file')


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()

    for step, (images, labels) in enumerate(train_queue):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(images)
        loss = criterion(logits, labels)
        if args.auxiliary:
            loss += args.auxiliary_weight * criterion(logits_aux, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, labels, topk=(1, 5))
        objs.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

        if step % args.report_freq == 0:
            logging.info('train %03d %.4e %.4f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (images, labels) in enumerate(valid_queue):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            logits, _ = model(images)
            loss = criterion(logits, labels)

            prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
            objs.update(loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            if step % args.report_freq == 0:
                logging.info('valid %03d %.4e %.4f %.4f',
                             step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        logging.error('No GPU available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
    logging.info('args = %s', args)

    genotype = _load_genotype()
    logging.info('Genotype: %s', genotype)

    num_classes = 100 if args.dataset == 'cifar100' else 10
    model = Network(args.init_channels, num_classes, args.layers,
                    args.auxiliary, genotype)
    model = model.cuda()
    logging.info('Params: %.2f MB', utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs))

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    DatasetClass = dset.CIFAR100 if args.dataset == 'cifar100' else dset.CIFAR10
    train_data = DatasetClass(root=args.data, train=True,
                              download=True, transform=train_transform)
    valid_data = DatasetClass(root=args.data, train=False,
                              download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=2)

    best_acc = 0.0
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %.4e', epoch, lr)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_loss = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %.4f', train_acc)

        valid_acc, valid_acc5, valid_loss = infer(valid_queue, model, criterion)
        logging.info('valid_acc %.4f  top5 %.4f', valid_acc, valid_acc5)

        if valid_acc > best_acc:
            best_acc = valid_acc
            utils.save(model, os.path.join(args.save, 'best_weights.pt'))
        utils.save(model, os.path.join(args.save, 'weights.pt'))

    logging.info('Best validation accuracy: %.4f%%', best_acc)
    logging.info('Test error: %.4f%%', 100.0 - best_acc)


if __name__ == '__main__':
    main()
