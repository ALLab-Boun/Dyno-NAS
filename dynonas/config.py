import argparse


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser('Dynonas search')
    p.add_argument('--data', default='../data',
                   help='path to dataset root')
    p.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                   help='search dataset')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--init_channels', type=int, default=16,
                   help='initial channel count for search network (8-cell stack)')
    p.add_argument('--layers', type=int, default=8,
                   help='total cells in the search network')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.025,
                   help='supernet SGD initial LR')
    p.add_argument('--lr_min', type=float, default=0.001,
                   help='cosine-annealing minimum LR')
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=3e-4)
    p.add_argument('--grad_clip', type=float, default=5.0)
    p.add_argument('--train_portion', type=float, default=0.5,
                   help='fraction of train data used for weight training; '
                        'remainder is the validation split used for fitness')
    p.add_argument('--arch_lr', type=float, default=3e-4,
                   help='Adam LR for arch parameters (alpha_D)')
    p.add_argument('--arch_wd', type=float, default=1e-3,
                   help='Adam weight decay for arch parameters')
    p.add_argument('--unrolled', action='store_true', default=False,
                   help='use second-order (unrolled) gradient for arch update')
    p.add_argument('--pop_size', type=int, default=50,
                   help='population size N')
    p.add_argument('--G_max', type=int, default=50,
                   help='maximum generations')
    p.add_argument('--tournament_size', type=int, default=20,
                   help='tournament size T for selection')
    p.add_argument('--crossover_prob', type=float, default=0.5,
                   help='crossover rate t_c (Algorithm 1)')
    p.add_argument('--mutation_prob', type=float, default=0.1,
                   help='mutation rate t_m (Algorithm 1). ')
    p.add_argument('--gradient_interval', type=int, default=5,
                   help='apply DARTS local search every n generations')
    p.add_argument('--mutation_type', default='default',
                   choices=['default', 'soft', 'adaptive', 'zero', 'hybrid'],
                   help='mutation operator: '
                        'default=standard '
                        'soft'
                        'adaptive '
                        'zero'
                        'hybrid=dynamic')
    p.add_argument('--disable_local_search', action='store_true', default=False,
                   help='ablation: skip DARTS local search')
    p.add_argument('--disable_crossover', action='store_true', default=False,
                   help='ablation: skip crossover (random parent choice)')
    p.add_argument('--disable_mutation', action='store_true', default=False,
                   help='ablation: skip mutation')
    p.add_argument('--local_search_variant', default='darts',
                   choices=['darts', 'beta_darts', 'shapley'],
                   help='local search method (darts is default; others for ablation)')
    p.add_argument('--cutout', action='store_true', default=False,
                   help='use cutout augmentation during search (off by default)')
    p.add_argument('--cutout_length', type=int, default=16)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save', default='search-dynonas',
                   help='experiment directory prefix')
    p.add_argument('--report_freq', type=int, default=10,
                   help='log every N generations')
    p.add_argument('--debug', action='store_true', default=False,
                   help='fast CPU smoke test: 2 generations, pop=4, 2 batches')

    return p


def get_args(argv=None):
    args = get_parser().parse_args(argv)
    if args.debug:
        args.pop_size = 4
        args.G_max = 2
        args.gradient_interval = 1
        args.batch_size = 16
        args.init_channels = 4
        args.layers = 2
        args.report_freq = 1
    return args
