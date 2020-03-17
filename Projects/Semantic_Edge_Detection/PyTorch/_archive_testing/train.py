# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
# from config import cfg
from dataset import TrainDataset
from models import models #ModelBuilder, SegmentationModule
from utils import AverageMeter, parse_devices, setup_logger
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback


# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # forward pass
        loss, acc = segmentation_module(batch_data[0])
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())


def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus):
    # Network Builders
    net_encoder = models.ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = models.ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)

    crit = nn.NLLLoss(ignore_index=-1)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = models.SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = models.SegmentationModule(
            net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    # segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(segmentation_module, iterator_train, optimizers, history, epoch+1, cfg)

        # checkpointing
        checkpoint(nets, history, cfg, epoch+1)

    print('Training Done!')


if __name__ == '__main__':
    # assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
    #     'PyTorch>=0.4.0 is required'

    # parser = argparse.ArgumentParser(
    #     description="PyTorch Semantic Segmentation Training"
    # )
    # parser.add_argument(
    #     "--cfg",
    #     default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
    #     metavar="FILE",
    #     help="path to config file",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--gpus",
    #     default="0-3",
    #     help="gpus to use, e.g. 0-3 or 0,1,2,3"
    # )
    # parser.add_argument(
    #     "opts",
    #     help="Modify config options using the command-line",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )
    # args = parser.parse_args()

    # cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)
    # # cfg.freeze()

    # logger = setup_logger(distributed_rank=0)   # TODO
    # logger.info("Loaded configuration file {}".format(args.cfg))
    # logger.info("Running with config:\n{}".format(cfg))

    # # Output directory
    # if not os.path.isdir(cfg.DIR):
    #     os.makedirs(cfg.DIR)
    # logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    # with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
    #     f.write("{}".format(cfg))

    # # Start from checkpoint
    # if cfg.TRAIN.start_epoch > 0:
    #     cfg.MODEL.weights_encoder = os.path.join(
    #         cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
    #     cfg.MODEL.weights_decoder = os.path.join(
    #         cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
    #     assert os.path.exists(cfg.MODEL.weights_encoder) and \
    #         os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # # Parse gpu ids
    # gpus = parse_devices(args.gpus)
    # gpus = [x.replace('gpu', '') for x in gpus]
    # gpus = [int(x) for x in gpus]
    # num_gpus = len(gpus)
    # cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    # cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    # cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    # cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    # random.seed(cfg.TRAIN.seed)
    # torch.manual_seed(cfg.TRAIN.seed)

    # main(cfg, gpus)

	assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
	    'PyTorch>=0.4.0 is required'
	from yacs.config import CfgNode as CN
	# -----------------------------------------------------------------------------
	# Config definition
	# -----------------------------------------------------------------------------

	_C = CN()
	_C.DIR = "ckpt/resnet50-upernet"

	# -----------------------------------------------------------------------------
	# Dataset
	# -----------------------------------------------------------------------------
	_C.DATASET = CN()
	_C.DATASET.root_dataset = "./data/"
	_C.DATASET.list_train = "./data/training.odgt"
	_C.DATASET.list_val = "./data/validation.odgt"
	_C.DATASET.num_class = 2
	# multiscale train/test, size of short edge (int or tuple)
	_C.DATASET.imgSizes = (300, 375, 450, 525, 600)
	# maximum input image size of long edge
	_C.DATASET.imgMaxSize = 1000
	# maxmimum downsampling rate of the network
	_C.DATASET.padding_constant = 32
	# downsampling rate of the segmentation label
	_C.DATASET.segm_downsampling_rate = 4
	# randomly horizontally flip images when train/test
	_C.DATASET.random_flip = True

	# -----------------------------------------------------------------------------
	# Model
	# -----------------------------------------------------------------------------
	_C.MODEL = CN()
	# architecture of net_encoder
	_C.MODEL.arch_encoder = "resnet50"
	# architecture of net_decoder
	_C.MODEL.arch_decoder = "upernet"
	# weights to finetune net_encoder
	_C.MODEL.weights_encoder = ""
	# weights to finetune net_decoder
	_C.MODEL.weights_decoder = ""
	# number of feature channels between encoder and decoder
	_C.MODEL.fc_dim = 2048

	# -----------------------------------------------------------------------------
	# Training
	# -----------------------------------------------------------------------------
	_C.TRAIN = CN()
	_C.TRAIN.batch_size_per_gpu = 1
	# epochs to train for
	_C.TRAIN.num_epoch = 1
	# epoch to start training. useful if continue from a checkpoint
	_C.TRAIN.start_epoch = 0
	# iterations of each epoch (irrelevant to batch size)
	_C.TRAIN.epoch_iters = 50

	_C.TRAIN.optim = "SGD"
	_C.TRAIN.lr_encoder = 0.02
	_C.TRAIN.lr_decoder = 0.02
	# power in poly to drop LR
	_C.TRAIN.lr_pow = 0.9
	# momentum for sgd, beta1 for adam
	_C.TRAIN.beta1 = 0.9
	# weights regularizer
	_C.TRAIN.weight_decay = 1e-4
	# the weighting of deep supervision loss
	_C.TRAIN.deep_sup_scale = 0.4
	# fix bn params, only under finetuning
	_C.TRAIN.fix_bn = False
	# number of data loading workers
	_C.TRAIN.workers = 1

	# frequency to display
	_C.TRAIN.disp_iter = 2
	# manual seed
	_C.TRAIN.seed = 304

	# -----------------------------------------------------------------------------
	# Validation
	# -----------------------------------------------------------------------------
	_C.VAL = CN()
	# currently only supports 1
	_C.VAL.batch_size = 1
	# output visualization during validation
	_C.VAL.visualize = False
	# the checkpoint to evaluate on
	_C.VAL.checkpoint = "epoch_30.pth"

	# -----------------------------------------------------------------------------
	# Testing
	# -----------------------------------------------------------------------------
	_C.TEST = CN()
	# currently only supports 1
	_C.TEST.batch_size = 1
	# the checkpoint to test on
	_C.TEST.checkpoint = "epoch_30.pth"
	# folder to output visualization results
	_C.TEST.result = "./"


	cfg = _C

	parser = argparse.ArgumentParser(
	    description="PyTorch Semantic Segmentation Training"
	)
	parser.add_argument(
	    "--cfg",
	    default="configuration/resnet50dilated-ppm_deepsup.yaml",
	    metavar="FILE",
	    help="path to config file",
	    type=str,
	)
	parser.add_argument(
	    "--gpus",
	    default="0",
	    help="gpus to use, e.g. 0-3 or 0,1,2,3"
	)
	parser.add_argument(
	    "opts",
	    help="Modify config options using the command-line",
	    default=None,
	    nargs=argparse.REMAINDER,
	)
	args = parser.parse_args(args=[])

	# cfg.merge_from_file(args.cfg)
	cfg.merge_from_list(args.opts)
	
	if not os.path.isdir(cfg.DIR):
		os.makedirs(cfg.DIR)

	with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
		f.write("{}".format(cfg))
    
    # Start from checkpoint
	if cfg.TRAIN.start_epoch > 0:
		cfg.MODEL.weights_encoder = os.path.join(
			cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
		cfg.MODEL.weights_decoder = os.path.join(
			cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
		assert os.path.exists(cfg.MODEL.weights_encoder) and \
			os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
	gpus = parse_devices(args.gpus)
	gpus = [x.replace('gpu', '') for x in gpus]
	gpus = [int(x) for x in gpus]
	num_gpus = len(gpus)
	cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

	cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
	cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
	cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

	random.seed(cfg.TRAIN.seed)
	torch.manual_seed(cfg.TRAIN.seed)

	main(cfg, gpus)
