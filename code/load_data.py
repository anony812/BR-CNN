import sys
import glob
import os

import numpy as np
import torch
from torchvision import transforms
import torchvision

from PIL import Image

import time
import args
import utils
import torch.distributed as dist
from random import Random
import scipy.io
import imageDatasetWithSeg
import fineGrainedDataset
from torch.utils.data.sampler import Sampler

class RandomSampler(Sampler):

    def __init__(self, data_source,seed):
        self.data_source = data_source
        np.random.seed(1)
        torch.manual_seed(1)

        self.randPerm = torch.randperm(len(self.data_source))
        self.iterObj = iter(self.randPerm.tolist())

    def __iter__(self):
        return self.iterObj

    def __len__(self):
        return self.num_samples

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def buildTrainLoader(args,transf=None,shuffle=True,withSeg=False,reprVec=False):

    if args.very_big_images:
        imgSize = 1792
    elif args.big_images:
        imgSize = 448
    else:
        imgSize = 224

    train_dataset = fineGrainedDataset.FineGrainedDataset(args.dataset_train, "train",(imgSize,imgSize),\
                                            withSeg=withSeg,sqResizing=args.sq_resizing,\
                                            cropRatio=args.crop_ratio,brightness=args.brightness,\
                                            saturation=args.saturation,withSaliency=args.saliency_crop,\
                                            randomSalCrop=args.random_sal_crop,apply_random_crop=args.apply_random_crop)

    totalLength = len(train_dataset)

    if args.prop_set_int_fmt:
        train_prop = args.train_prop / 100
    else:
        train_prop = args.train_prop

    np.random.seed(1)
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)

    train_dataset, _ = torch.utils.data.random_split(train_dataset, [int(totalLength * train_prop),
                                                                     totalLength - int(totalLength * train_prop)])

    kwargs = {"shuffle": shuffle}

    if args.distributed:
        size = dist.get_world_size()
        bsz = int(args.batch_size / float(size))
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes)
        partition = partition.use(dist.get_rank())
    else:
        bsz = args.batch_size

    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bsz,  # use custom collate function here
                                              pin_memory=False, num_workers=args.num_workers, **kwargs)

    return trainLoader, train_dataset


def buildTestLoader(args, mode,shuffle=False,withSeg=False,reprVec=False):
    datasetName = getattr(args, "dataset_{}".format(mode))
    imgSize = 448 if args.big_images else 224
    test_dataset = fineGrainedDataset.FineGrainedDataset(datasetName, mode,(imgSize,imgSize),\
                                                        withSeg=withSeg,sqResizing=args.sq_resizing,\
                                                        cropRatio=args.crop_ratio,brightness=args.brightness,saturation=args.saturation)

    if mode == "val" and args.dataset_train == args.dataset_val:
        np.random.seed(1)
        torch.manual_seed(1)
        if args.cuda:
            torch.cuda.manual_seed(1)

        if args.prop_set_int_fmt:
            train_prop = args.train_prop / 100
        else:
            train_prop = args.train_prop

        totalLength = len(test_dataset)
        _, test_dataset = torch.utils.data.random_split(test_dataset, [int(totalLength * train_prop),
                                                                       totalLength - int(totalLength * train_prop)])

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    if shuffle:
        sampler = RandomSampler(test_dataset,args.seed)
    else:
        sampler=None

    if args.val_batch_size == -1:
        args.val_batch_size = int(args.max_batch_size*3.5)

    testLoader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.val_batch_size,
                                             num_workers=args.num_workers,sampler=sampler)

    return testLoader,test_dataset


def addArgs(argreader):
    argreader.parser.add_argument('--pretrain_dataset', type=str, metavar='N',
                                  help='The network producing the features can only be pretrained on \'imageNet\'. This argument must be \
                            set to \'imageNet\' datasets.')
    argreader.parser.add_argument('--batch_size', type=int, metavar='BS',
                                  help='The batchsize to use for training')
    argreader.parser.add_argument('--val_batch_size', type=int, metavar='BS',
                                  help='The batchsize to use for validation')
    argreader.parser.add_argument('--max_sub_batch_size', type=int, metavar='BS',
                                  help='The maximum sub batch size when using very big images.')

    argreader.parser.add_argument('--train_prop', type=float, metavar='END',
                                  help='The proportion of the train dataset to use for training when working in non video mode. The rest will be used for validation.')

    argreader.parser.add_argument('--prop_set_int_fmt', type=args.str2bool, metavar='BOOL',
                                  help='Set to True to set the sets (train, validation and test) proportions\
                            using int between 0 and 100 instead of float between 0 and 1.')

    argreader.parser.add_argument('--dataset_train', type=str, metavar='DATASET',
                                  help='The dataset for training. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_val', type=str, metavar='DATASET',
                                  help='The dataset for validation. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_test', type=str, metavar='DATASET',
                                  help='The dataset for testing. Can be "big" or "small"')

    argreader.parser.add_argument('--shuffle_test_set', type=args.str2bool, metavar='BOOL',
                                  help='To shuffle the test set.')


    argreader.parser.add_argument('--class_nb', type=int, metavar='S',
                                  help='The number of class of to model')

    argreader.parser.add_argument('--old_preprocess', type=args.str2bool, metavar='S',
                                  help='To use the old images pre-processor.')
    argreader.parser.add_argument('--moredataaug_preprocess', type=args.str2bool, metavar='S',
                                  help='To apply color jitter and random rotation along random resized crop and horizontal flip')
    argreader.parser.add_argument('--ws_dan_preprocess', type=args.str2bool, metavar='S',
                                  help='To apply the same image preprocessing as in WS-DAN.')

    argreader.parser.add_argument('--upscale_test', type=args.str2bool, metavar='S',
                                  help='To increase test resolution from 224 to 312')

    argreader.parser.add_argument('--big_images', type=args.str2bool, metavar='S',
                                  help='To resize the images to 448 pixels instead of 224')
    argreader.parser.add_argument('--very_big_images', type=args.str2bool, metavar='S',
                                    help='To resize the images to 1792 pixels instead of 224')


    argreader.parser.add_argument('--normalize_data', type=args.str2bool, metavar='S',
                                  help='To normalize the data using imagenet means and std before puting it between 0 and 1.')

    argreader.parser.add_argument('--with_seg', type=args.str2bool, metavar='S',
                                  help='To load segmentation along with image and target')

    argreader.parser.add_argument('--repr_vec', type=args.str2bool, metavar='S',
                                  help='To use representative vectors instead of raw image.')

    argreader.parser.add_argument('--sq_resizing', type=args.str2bool, metavar='S',
                                  help='To resize each input image to a square.')

    argreader.parser.add_argument('--crop_ratio', type=float, metavar='S',
                                  help='The crop ratio (usually 0.875) for data augmentation.')
    argreader.parser.add_argument('--brightness', type=float, metavar='S',
                                  help='The brightness intensity for data augmentation.')
    argreader.parser.add_argument('--saturation', type=float, metavar='S',
                                  help='The saturation intensity for data augmentation.')

    argreader.parser.add_argument('--saliency_crop', type=args.str2bool, metavar='S',
                                  help='To crop image using saliency.')

    argreader.parser.add_argument('--random_sal_crop', type=args.str2bool, metavar='S',
                                  help='To do random saliency cropping.')

    argreader.parser.add_argument('--apply_random_crop', type=args.str2bool, metavar='S',
                                  help='To apply random crop after saliency crop.')

    return argreader
