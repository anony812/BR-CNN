import utils
import numpy as np
import torch
import scipy as sp
import torch
import sys
import torch.nn.functional as F
import math

import load_data
import torchvision

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def updateMetrDict(metrDict,metrDictSample):

    if metrDict is None:
        metrDict = metrDictSample
    else:
        for metric in metrDict.keys():
            metrDict[metric] += metrDictSample[metric]

    return metrDict

def binaryToMetrics(output,target,segmentation,resDict,comp_spars=False):
    ''' Computes metrics over a batch of targets and predictions

    Args:
    - output (list): the batch of outputs
    - target (list): the batch of ground truth class
    - transition_matrix (torch.tensor) : this matrix contains at row i and column j the empirical probability to go from state i to j

    '''

    acc = compAccuracy(output,target)
    metDict = {"Accuracy":acc}

    for key in resDict.keys():
        if key.find("pred_") != -1:
            suff = key.split("_")[-1]
            metDict["Accuracy_{}".format(suff)] = compAccuracy(resDict[key],target)

    if "attMaps" in resDict.keys() and comp_spars:
        segmentation = segmentation.clone()

        spar,spar_n,ios = compAttMapSparsity(resDict["attMaps"].clone(),resDict["features"].clone(),segmentation)
        metDict["Sparsity"],metDict["Sparsity Normalised"] = spar,spar_n
        metDict["IoS"] = ios

    return metDict

def compAccuracy(output,target):
    pred = output.argmax(dim=-1)
    acc = (pred == target).float().sum()
    return acc.item()

def compAttMapSparsity(attMaps,features=None,segmentation=None):
    if not features is None:
        norm = torch.sqrt(torch.pow(features,2).sum(dim=1,keepdim=True))
        norm_max = norm.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
        norm = norm/norm_max

        attMaps = attMaps*norm

    max = attMaps.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    attMaps = attMaps/(max+0.00001)

    if attMaps.size(1) > 1:
        attMaps = attMaps.mean(dim=1,keepdim=True)

    sparsity = attMaps.mean(dim=(2,3))

    factor = segmentation.size(-1)/attMaps.size(-1)
    sparsity_norm = sparsity/((segmentation>0.5).sum(dim=(2,3)).sum(dim=1,keepdim=True)/factor).float()
    ios = compIoS(attMaps,segmentation)

    return sparsity.sum().item(),sparsity_norm.sum().item(),ios.sum().item()

def compIoS(attMapNorm,segmentation):

    segmentation = (segmentation>0.5)

    thresholds = torch.arange(10)*1.0/10

    attMapNorm = F.interpolate(attMapNorm,size=(segmentation.size(-1)),mode="bilinear",align_corners=False)

    allIos = []

    for thres in thresholds:
        num = ((attMapNorm>thres)*segmentation[:,0:1]).sum(dim=(1,2,3)).float()
        denom = (attMapNorm>thres).sum(dim=(1,2,3)).float()
        ios = num/denom
        ios[torch.isnan(ios)] = 0
        allIos.append(ios.unsqueeze(0))

    finalIos = torch.cat(allIos,dim=0).mean(dim=0)
    return finalIos
