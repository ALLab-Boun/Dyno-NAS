from __future__ import annotations

import os
import sys
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset

# Ensure cnn/ is importable from exp_final/genas/
_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, '..'))
_CNN  = os.path.join(_ROOT, 'cnn')
for _p in [_ROOT, _CNN]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import utils                                  # cnn/utils.py
import genotypes as _genotypes_module         # cnn/genotypes.py
from genotypes import Genotype                # needed for eval() when loading from file
from model import NetworkCIFAR as Network     # cnn/model.py


def get_args(argv=None):
    p = argparse.ArgumentParser('dynonas test-set evaluation')

    # Data
    p.add_argument('--data', default='./data',
                   help='path to dataset root')
    p.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    p.add_argument('--batch_size', type=int, default=96)
    p.add_argument('--num_workers', type=int, default=2)

    # Checkpoint
    p.add_argument('--model_path', required=True,
                   help='path to best_weights.pt saved by train_final.py')

    # Architecture — one of the two must be provided
    p.add_argument('--genotype_file', default=None,
                   help='path to best_genotype.txt produced by dynonas_search.py')
    p.add_argument('--arch', default=None,
                   help='named genotype from cnn/genotypes.py (e.g. DARTS_V2)')

    # Model structure (must match what was used during training)
    p.add_argument('--init_channels', type=int, default=36)
    p.add_argument('--layers', type=int, default=20)
    p.add_argument('--auxiliary', action='store_true', default=False,
                   help='model was trained with auxiliary tower')
    p.add_argument('--drop_path_prob', type=float, default=0.0,
                   help='set >0 to apply drop-path during test (usually 0 for eval)')
    p.add_argument('--cutout', action='store_true', default=False,
                   help='apply cutout augmentation (match training setting)')
    p.add_argument('--cutout_length', type=int, default=16)

    # Misc
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--no_cuda', action='store_true', default=False,
                   help='force CPU evaluation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--report_freq', type=int, default=50)

    args = p.parse_args(argv)

    if args.genotype_file is None and args.arch is None:
        p.error('Provide --genotype_file or --arch')

    return args


def _load_genotype(args):
    if args.genotype_file:
        with open(args.genotype_file) as f:
            return eval(f.read().strip())
    return getattr(_genotypes_module, args.arch)


def _infer(test_queue, model, criterion, device, report_freq):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (images, labels) in enumerate(test_queue):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _ = model(images)
            loss = criterion(logits, labels)

            prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
            n = images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % report_freq == 0:
                logging.info('test %03d | loss %.4e | top-1 %.2f%% | top-5 %.2f%%',
                             step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


def main():
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S %p')

    args = get_args()
    logging.info('args = %s', args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda', args.gpu)
        logging.info('Device: GPU %d', args.gpu)
    else:
        device = torch.device('cpu')
        logging.info('Device: CPU')

    # ---- Genotype -----------------------------------------------------------
    genotype = _load_genotype(args)
    logging.info('Genotype: %s', genotype)

    # ---- Model --------------------------------------------------------------
    # Auto-detect auxiliary head from checkpoint so the user doesn't need to
    # remember to pass --auxiliary when the model was trained with it.
    checkpoint = torch.load(args.model_path, map_location='cpu')
    has_auxiliary = any(k.startswith('auxiliary_head.') for k in checkpoint.keys())
    if has_auxiliary and not args.auxiliary:
        logging.warning('Checkpoint contains auxiliary_head weights — '
                        'enabling --auxiliary automatically.')
        args.auxiliary = True

    num_classes = 100 if args.dataset == 'cifar100' else 10
    model = Network(args.init_channels, num_classes, args.layers,
                    args.auxiliary, genotype)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.drop_path_prob = args.drop_path_prob
    logging.info('Params: %.2f MB', utils.count_parameters_in_MB(model))
    logging.info('Checkpoint: %s', args.model_path)

    # ---- Data ---------------------------------------------------------------
    _, test_transform = utils._data_transforms_cifar10(args)
    DatasetClass = dset.CIFAR100 if args.dataset == 'cifar100' else dset.CIFAR10
    test_data = DatasetClass(root=args.data, train=False,
                             download=True, transform=test_transform)
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_cuda, num_workers=args.num_workers)

    # ---- Evaluation ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss().to(device)
    top1, top5, loss = _infer(test_queue, model, criterion, device, args.report_freq)

    logging.info('=' * 50)
    logging.info('Dataset  : %s', args.dataset.upper())
    logging.info('Top-1 acc: %.2f%%', top1)
    logging.info('Top-5 acc: %.2f%%', top5)
    logging.info('Top-1 err: %.2f%%', 100.0 - top1)
    logging.info('Test loss: %.4e', loss)
    logging.info('=' * 50)


if __name__ == '__main__':
    main()
