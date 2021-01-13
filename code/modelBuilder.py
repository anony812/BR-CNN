import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel

import args
import sys

import glob
from skimage.transform import resize
import matplotlib.pyplot as plt

from models import deeplab
from models import resnet
from models import bagnet
from models import hrnet
import torchvision

try:
    import torch_geometric
except ModuleNotFoundError:
    pass

import skimage.feature
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import cv2
from scipy import ndimage

from  torch.nn.modules.upsampling import Upsample
import time

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import random
import math

def buildFeatModel(featModelName, pretrainedFeatMod, featMap=True, bigMaps=False, layerSizeReduce=False, stride=2,dilation=1,deeplabv3_outchan=64, **kwargs):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''
    if featModelName.find("deeplabv3") != -1:
        featModel = deeplab._segm_resnet("deeplabv3", featModelName[featModelName.find("resnet"):], \
                                         outChan=deeplabv3_outchan,\
                                         pretrained=pretrainedFeatMod, featMap=featMap, layerSizeReduce=layerSizeReduce,
                                         **kwargs)
    elif featModelName.find("resnet") != -1:
        featModel = getattr(resnet, featModelName)(pretrained=pretrainedFeatMod, featMap=featMap,layerSizeReduce=layerSizeReduce, **kwargs)
    elif featModelName.find("bagnet") != -1:
        featModel = getattr(bagnet, featModelName)(pretrained=pretrainedFeatMod,strides=[2,2,2,1] if layerSizeReduce else [2,2,1,1], **kwargs)
    elif featModelName == "hrnet":
        featModel = hrnet.get_cls_net()
    elif featModelName == "hrnet64":
        featModel = hrnet.get_cls_net(w=64)
    elif featModelName == "hrnet18":
        featModel = hrnet.get_cls_net(w=18)
    else:
        raise ValueError("Unknown model type : ", featModelName)

    return featModel

# This class is just the class nn.DataParallel that allow running computation on multiple gpus
# but it adds the possibility to access the attribute of the model
class DataParallelModel(nn.DataParallel):
    def __init__(self, model):
        super(DataParallelModel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(DataParallelModel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Model(nn.Module):

    def __init__(self, firstModel, secondModel,nbFeat=512,drop_and_crop=False,zoom=False,zoom_max_sub_clouds=2,zoom_merge_preds=False,reducedImgSize=1,\
                        upscaledTest=False):
        super(Model, self).__init__()
        self.firstModel = firstModel
        self.secondModel = secondModel
        self.zoom = zoom
        self.reducedImgSize = reducedImgSize
        self.subcloudNb = zoom_max_sub_clouds
        self.zoom_merge_preds = zoom_merge_preds
        self.nbFeat = nbFeat
        self.drop_and_crop = drop_and_crop
        if drop_and_crop:
            self.bn = nn.BatchNorm2d(nbFeat, eps=0.001)
        self.upscaledTest = upscaledTest

    def forward(self, origImgBatch):

        if not self.firstModel is None:

            visResDict = self.firstModel(origImgBatch)
            resDict = self.secondModel(visResDict)
            resDict = merge(visResDict,resDict)

        else:
            resDict = self.secondModel(origImgBatch)

        return resDict

def merge(dictA,dictB,suffix=""):
    for key in dictA.keys():
        if key in dictB:
            dictB[key+"_"+suffix] = dictA[key]
        else:
            dictB[key] = dictA[key]
    return dictB

################################# Visual Model ##########################

class FirstModel(nn.Module):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False, **kwargs):
        super(FirstModel, self).__init__()

        self.featMod = buildFeatModel(featModelName, pretrainedFeatMod, featMap, bigMaps, **kwargs)

        self.featMap = featMap
        self.bigMaps = bigMaps

    def forward(self, x):
        raise NotImplementedError

class CNN2D(FirstModel):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False,aux_model=False,**kwargs):
        super(CNN2D, self).__init__(featModelName, pretrainedFeatMod, featMap, bigMaps,**kwargs)

        self.aux_model= aux_model

    def forward(self, x):

        # N x C x H x L
        self.batchSize = x.size(0)

        # N x C x H x L
        featModRetDict = self.featMod(x)
        features = featModRetDict["x"]

        spatialWeights = torch.pow(features, 2).sum(dim=1, keepdim=True)
        retDict = {}

        if not "attMaps" in featModRetDict.keys():
            retDict["attMaps"] = spatialWeights
            retDict["features"] = features
        else:
            retDict["attMaps"] = featModRetDict["attMaps"]
            retDict["features"] = featModRetDict["features"]

        retDict["x"] = features.mean(dim=-1).mean(dim=-1)

        if self.aux_model:
            retDict["auxFeat"] = features

        return retDict

def buildImageAttention(inFeat,outChan=1):
    attention = []
    attention.append(resnet.BasicBlock(inFeat, inFeat))
    attention.append(resnet.conv1x1(inFeat, outChan))
    return nn.Sequential(*attention)

class SoftMax(nn.Module):
    def __init__(self,norm=True,dim=-1):
        super(SoftMax,self).__init__()
        self.norm = norm
        self.dim = dim
    def forward(self,x):
        if self.dim == -1:
            origSize = x.size()
            x = torch.softmax(x.view(x.size(0),-1),dim=-1).view(origSize)
        elif self.dim == 1:
            x = torch.softmax(x.permute(0,2,3,1),dim=-1).permute(0,3,1,2)
        if self.norm:
            x_min,x_max = x.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0],x.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
            x = (x-x_min)/(x_max-x_min)
        return x

def representativeVectors(x,nbVec,applySoftMax=False,softmCoeff=1,softmSched=False,softmSched_interpCoeff=0,no_refine=False,randVec=False,unnorm=False,update_sco_by_norm_sim=False,vectIndToUse="all"):

    xOrigShape = x.size()

    normNotFlat = torch.sqrt(torch.pow(x,2).sum(dim=1,keepdim=True))

    x = x.permute(0,2,3,1).reshape(x.size(0),x.size(2)*x.size(3),x.size(1))
    norm = torch.sqrt(torch.pow(x,2).sum(dim=-1)) + 0.00001

    if randVec:
        raw_reprVec_score = torch.rand(norm.size()).to(norm.device)
    else:
        raw_reprVec_score = norm.clone()

    repreVecList = []
    simList = []
    for i in range(nbVec):
        _,ind = raw_reprVec_score.max(dim=1,keepdim=True)
        raw_reprVec_norm = norm[torch.arange(x.size(0)).unsqueeze(1),ind]
        raw_reprVec = x[torch.arange(x.size(0)).unsqueeze(1),ind]
        sim = (x*raw_reprVec).sum(dim=-1)/(norm*raw_reprVec_norm)

        if applySoftMax:
            if softmSched:
                simNorm = (1-softmSched_interpCoeff)*(sim/sim.sum(dim=1,keepdim=True))+softmSched_interpCoeff*torch.softmax(softmCoeff*sim,dim=1)
            else:
                simNorm = torch.softmax(softmCoeff*sim,dim=1)
        else:
            simNorm = sim/sim.sum(dim=1,keepdim=True)

        if unnorm:
            reprVec = (x*(simNorm*norm).unsqueeze(-1)).sum(dim=1)
        else:
            reprVec = (x*simNorm.unsqueeze(-1)).sum(dim=1)

        if not no_refine:
            repreVecList.append(reprVec)
        else:
            repreVecList.append(raw_reprVec[:,0])

        if randVec:
            raw_reprVec_score = torch.rand(norm.size()).to(norm.device)
        else:
            if update_sco_by_norm_sim:
                raw_reprVec_score = (1-simNorm)*raw_reprVec_score
            else:
                raw_reprVec_score = (1-sim)*raw_reprVec_score

        simReshaped = simNorm.reshape(sim.size(0),1,xOrigShape[2],xOrigShape[3])

        simList.append(simReshaped)

    if vectIndToUse == "all":
        return repreVecList,simList
    else:
        vectIndToUse = [int(ind) for ind in vectIndToUse.split(",")]
        return [repreVecList[ind] for ind in vectIndToUse],[simList[ind] for ind in vectIndToUse]

class CNN2D_bilinearAttPool(FirstModel):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False, chan=64,
                 inFeat=512,nb_parts=3,aux_model=False,score_pred_act_func="softmax",center_loss=False,\
                 center_loss_beta=5e-2,num_classes=200,cuda=True,cluster=False,cluster_ensemble=False,applySoftmaxOnSim=False,\
                 softmCoeff=1,softmSched=False,normFeat=False,no_refine=False,rand_vec=False,unnorm=False,update_sco_by_norm_sim=False,\
                 vect_gate=False,vect_ind_to_use="all",multi_feat_by_100=False,cluster_lay_ind=4,clu_glob_vec=False,\
                 clu_glob_rep_vec=False,clu_glob_corr_vec=False,clus_glob_norefine=False,applySoftmaxOnSim_glob=False,\
                 clu_deconv=False,\
                 **kwargs):

        super(CNN2D_bilinearAttPool, self).__init__(featModelName, pretrainedFeatMod, featMap, bigMaps, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if not cluster:
            self.attention = buildImageAttention(inFeat,nb_parts+1)
        else:
            self.attention = None

        self.nb_parts = nb_parts
        self.normFeat = normFeat
        self.cluster = cluster
        if not cluster:
            if score_pred_act_func == "softmax":
                self.attention_activation = SoftMax(norm=False,dim=1)
            elif score_pred_act_func == "relu":
                self.attention_activation = torch.relu
            elif score_pred_act_func == "sigmoid":
                self.attention_activation = torch.sigmoid
            else:
                raise ValueError("Unkown activation function : ",score_pred_act_func)
        else:
            self.attention_activation = None
            self.cluster_ensemble = cluster_ensemble
            self.applySoftmaxOnSim = applySoftmaxOnSim
            self.no_refine = no_refine
            self.rand_vec = rand_vec
            self.unnorm = unnorm
            self.update_sco_by_norm_sim = update_sco_by_norm_sim

        self.aux_model = aux_model

        self.center_loss = center_loss
        if self.center_loss:
            self.center_loss_beta = center_loss_beta
            self.feature_center = torch.zeros(num_classes, nb_parts * inFeat)
            self.feature_center = self.feature_center.cuda() if cuda else self.feature_center

        self.softmSched = softmSched
        self.softmSched_interpCoeff = 0
        self.softmCoeff = softmCoeff

        self.vect_gate = vect_gate
        if self.vect_gate:
            self.vect_gate_proto = torch.nn.Parameter(torch.zeros(nb_parts,inFeat),requires_grad=True)
            stdv = 1. / math.sqrt(self.vect_gate_proto.size(1))
            self.vect_gate_proto.data.uniform_(0, 2*stdv)

        self.vect_ind_to_use = vect_ind_to_use
        self.multi_feat_by_100 = multi_feat_by_100

        self.cluster_lay_ind = cluster_lay_ind
        self.clu_glob_vec = clu_glob_vec
        self.clu_glob_rep_vec = clu_glob_rep_vec
        self.clu_glob_corr_vec = clu_glob_corr_vec
        self.clus_glob_norefine = clus_glob_norefine
        self.applySoftmaxOnSim_glob = applySoftmaxOnSim_glob

        self.clu_deconv = clu_deconv
        if self.clu_deconv and featModelName != "resnet50":
            raise ValueError("Deconv only implemented for resnet50.")
        if self.clu_deconv:
            self.deconv = resnet.revresnet50()

            if cuda:
                self.deconv.cuda()

            if isinstance(self.featMod,DataParallelModel):
                self.deconv = DataParallelModel(self.deconv)

    def updateDeconv(self):
        params = self.featMod.state_dict()
        self.deconv.load_state_dict(params,strict=False)

    def forward(self, x):
        # N x C x H x L
        self.batchSize = x.size(0)
        # N x C x H x L
        output = self.featMod(x)

        if (not self.cluster) or (self.cluster_lay_ind == 4):
            features = output["x"]
        else:
            features = output["layerFeat"][self.cluster_lay_ind]

        if self.normFeat:
            features = features/torch.sqrt(torch.pow(features,2).sum(dim=1,keepdim=True))

        retDict = {}

        #features[:,:,:,:3] = 0
        #features[:,:,:,-3:] = 0

        if not self.cluster:
            spatialWeights = self.attention_activation(self.attention(features))
            features_weig = (spatialWeights[:,:self.nb_parts].unsqueeze(2)*features.unsqueeze(1)).reshape(features.size(0),features.size(1)*(spatialWeights.size(1)-1),features.size(2),features.size(3))
            features_agr = self.avgpool(features_weig)
            features_agr = features_agr.view(features.size(0), -1)
        else:

            vecList,simList = representativeVectors(features,self.nb_parts,self.applySoftmaxOnSim,self.softmCoeff,self.softmSched,\
                                                    self.softmSched_interpCoeff,self.no_refine,self.rand_vec,self.unnorm,\
                                                    self.update_sco_by_norm_sim,self.vect_ind_to_use)

            if not self.cluster_ensemble:
                if self.vect_gate:
                    features_agr = torch.cat(vecList,dim=0)

                    if self.vect_ind_to_use == "all":
                        features_agr = features_agr.unsqueeze(1).reshape(features_agr.size(0)//self.nb_parts,self.nb_parts,features_agr.size(1))
                    else:
                        effectivePartNb = len(self.vect_ind_to_use.split(","))
                        features_agr = features_agr.unsqueeze(1).reshape(features_agr.size(0)//effectivePartNb,effectivePartNb,features_agr.size(1))

                    # (N 1 3 512) x (1 3 1 512) -> (N 3 3 1)
                    sim = (features_agr.unsqueeze(1) * self.vect_gate_proto.unsqueeze(0).unsqueeze(2)).sum(dim=-1,keepdim=True)

                    featNorm = torch.sqrt(torch.pow(features_agr,2).sum(dim=-1,keepdim=True))
                    vect_gate_proto_norm = torch.sqrt(torch.pow(self.vect_gate_proto,2).sum(dim=-1,keepdim=True))

                    sim = sim/(featNorm.unsqueeze(2) * vect_gate_proto_norm.unsqueeze(0).unsqueeze(1))

                    # (N 1 3 512) x (N 3 3 1) -> (N 3 3 512) -> (N 3 512)
                    features_agr = (features_agr.unsqueeze(1) * torch.softmax(sim,dim=-2)).sum(dim=-2)
                    features_agr = features_agr.reshape(features_agr.size(0),-1)
                else:
                    features_agr = torch.cat(vecList,dim=-1)

            else:
                features_agr = vecList

            spatialWeights = torch.cat(simList,dim=1)

        if self.multi_feat_by_100:
            retDict["x"] = 100*features_agr
        else:
            retDict["x"] = features_agr

        if not self.clu_deconv or self.training:
            retDict["attMaps"] = spatialWeights
            retDict["features"] = features
        else:
            allAttMaps = []
            for i in range(spatialWeights.size(1)):
                maps = spatialWeights[:,i:i+1]
                feature_weig = features*maps
                feature_weig = self.deconv(feature_weig)["x"]
                attMap = torch.sqrt(torch.pow(feature_weig,2).sum(dim=1,keepdim=True))
                allAttMaps.append(attMap)
            allAttMaps = torch.cat(allAttMaps,dim=1)
            retDict["attMaps"] = allAttMaps
            retDict["features"] = output["layerFeat"][2]

        if self.clu_glob_vec:
            retDict["x"] = torch.cat((retDict["x"],output["layerFeat"][4].mean(dim=-1).mean(dim=-1)),dim=-1)
        elif self.clu_glob_rep_vec:
            lastLayFeat = output["layerFeat"][4]

            globVecList,globSimList = representativeVectors(lastLayFeat,self.nb_parts,self.applySoftmaxOnSim_glob,self.softmCoeff,self.softmSched,\
                                                    self.softmSched_interpCoeff,self.clus_glob_norefine,self.rand_vec,self.unnorm,\
                                                    self.update_sco_by_norm_sim,self.vect_ind_to_use)
            globRepVec = torch.cat(globVecList,dim=-1)
            retDict["x"] = torch.cat((retDict["x"],globRepVec),dim=-1)

            retDict["attMaps_glob"] = torch.cat(globSimList,dim=1)
            retDict["features_glob"] = lastLayFeat
        elif self.clu_glob_corr_vec:
            lastLayFeat = output["layerFeat"][4]

            spatialWeights_max = spatialWeights.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
            inds = torch.nonzero(spatialWeights==spatialWeights_max,as_tuple=False)

            #Removing vectors when they are too many (in case several of them have weight==1)
            if len(inds) > spatialWeights.size(0)*spatialWeights.size(1):

                indToKeep = [0]
                for i in range(1,len(inds)):
                    if (inds[i,:2] != inds[i-1,:2]).any():
                        indToKeep.append(i)
                inds = inds[indToKeep]

            x = (inds[:,3].float()*(lastLayFeat.size(3)*1.0/spatialWeights.size(3))).long()
            y = (inds[:,2].float()*(lastLayFeat.size(2)*1.0/spatialWeights.size(2))).long()

            globCorVec = lastLayFeat[inds[:,0],:,y,x]

            globCorVec = globCorVec.reshape(lastLayFeat.size(0),-1)

            retDict["x"] = torch.cat((retDict["x"],globCorVec),dim=-1)
            retDict["features_glob"] = lastLayFeat

        if self.center_loss:
            retDict["feature_matrix"] = retDict["x"]

        if self.aux_model:
            retDict["auxFeat"] = features

        return retDict

    def computeFeatCenter(self,target):
        return F.normalize(self.feature_center[target], dim=-1)
    def updateFeatCenter(self,feature_center_batch,features_agr,target):
        self.feature_center[target] += self.center_loss_beta * (features_agr.detach() - feature_center_batch)

################################ Temporal Model ########################""

class SecondModel(nn.Module):

    def __init__(self, nbFeat, nbClass):
        super(SecondModel, self).__init__()
        self.nbFeat, self.nbClass = nbFeat, nbClass

    def forward(self, x):
        raise NotImplementedError

class LinearSecondModel(SecondModel):

    def __init__(self, nbFeat, nbFeatAux,nbClass, dropout,aux_model=False,zoom=False,zoom_max_sub_clouds=2,bil_cluster_ensemble=False,\
                        bil_cluster_ensemble_gate=False,gate_drop=False,gate_randdrop=False,bias=True,\
                        aux_on_masked=False):

        super(LinearSecondModel, self).__init__(nbFeat, nbClass)
        self.dropout = nn.Dropout(p=dropout)

        self.linLay = nn.Linear(self.nbFeat, self.nbClass,bias=bias)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bil_cluster_ensemble = bil_cluster_ensemble

        self.aux_on_masked = aux_on_masked
        if self.aux_on_masked:
            self.lin01 = nn.Linear(int(nbFeat*2/3),nbClass)
            self.lin12 = nn.Linear(int(nbFeat*2/3),nbClass)
            self.lin0 = nn.Linear(nbFeat//3,nbClass)
            self.lin1 = nn.Linear(nbFeat//3,nbClass)
            self.lin2 = nn.Linear(nbFeat//3,nbClass)

    def forward(self, visResDict):

        if not self.bil_cluster_ensemble:
            x = visResDict["x"]

            if len(x.size()) == 4:
                x = self.avgpool(x).squeeze(-1).squeeze(-1)

            x = self.dropout(x)
            pred = self.linLay(x)

            retDict = {"pred": pred}

            if self.aux_on_masked:
                retDict["pred_01"] = self.lin01(x[:,:int(self.nbFeat*2/3)].detach())
                retDict["pred_12"] = self.lin12(x[:,int(self.nbFeat*1/3):].detach())
                retDict["pred_0"] = self.lin0(x[:,:int(self.nbFeat*1/3)].detach())
                retDict["pred_1"] = self.lin1(x[:,int(self.nbFeat*1/3):int(self.nbFeat*2/3)].detach())
                retDict["pred_2"] = self.lin2(x[:,int(self.nbFeat*2/3):].detach())

        else:
            predList = []
            gateScoreList = []

            for featVec in visResDict["x"]:
                predList.append(self.linLay(featVec).unsqueeze(0))

            x = torch.cat(predList,dim=0).mean(dim=0)

            retDict = {"pred": x}
            retDict.update({"predBilClusEns{}".format(i):predList[i][0] for i in range(len(predList))})

        return retDict

def getResnetFeat(backbone_name, backbone_inplanes,deeplabv3_outchan):
    if backbone_name == "resnet50" or backbone_name == "resnet101" or backbone_name == "resnet151":
        nbFeat = backbone_inplanes * 4 * 2 ** (4 - 1)
    elif backbone_name.find("deeplab") != -1:
        nbFeat = deeplabv3_outchan
    elif backbone_name.find("resnet34") != -1:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    elif backbone_name.find("resnet18") != -1:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    elif backbone_name.find("resnet14") != -1:
        nbFeat = backbone_inplanes * 2 ** (3 - 1)
    elif backbone_name.find("resnet9") != -1:
        nbFeat = backbone_inplanes * 2 ** (2 - 1)
    elif backbone_name.find("resnet4") != -1:
        nbFeat = backbone_inplanes * 2 ** (1 - 1)
    elif backbone_name.find("bagnet33") != -1:
        nbFeat = 2048
    elif backbone_name == "hrnet":
        nbFeat = 44
    elif backbone_name == "hrnet64":
        nbFeat = 64
    elif backbone_name == "hrnet18":
        nbFeat = 16
    else:
        raise ValueError("Unkown backbone : {}".format(backbone_name))
    return nbFeat

def netBuilder(args):
    ############### Visual Model #######################

    if not args.repr_vec:
        if args.first_mod.find("resnet") != -1 or args.first_mod.find("bagnet") != -1 or args.first_mod.find("hrnet") != -1:

            nbFeat = getResnetFeat(args.first_mod, args.resnet_chan,args.deeplabv3_outchan)

            if args.resnet_bilinear:
                CNNconst = CNN2D_bilinearAttPool
                kwargs = {"inFeat":nbFeat,"aux_model":args.aux_model,"nb_parts":args.resnet_bil_nb_parts,\
                          "score_pred_act_func":args.resnet_simple_att_score_pred_act_func,
                          "center_loss":args.bil_center_loss,"center_loss_beta":args.bil_center_loss_beta,\
                          "cuda":args.cuda,"cluster":args.bil_cluster,"cluster_ensemble":args.bil_cluster_ensemble,\
                          "applySoftmaxOnSim":args.apply_softmax_on_sim,\
                          "softmCoeff":args.softm_coeff,\
                          "softmSched":args.bil_clus_soft_sched,\
                          "no_refine":args.bil_cluster_norefine,\
                          "rand_vec":args.bil_cluster_randvec,\
                          "unnorm":args.bil_clust_unnorm,\
                          "update_sco_by_norm_sim":args.bil_clust_update_sco_by_norm_sim,\
                          "normFeat":args.bil_norm_feat,\
                          "vect_gate":args.bil_clus_vect_gate,\
                          "vect_ind_to_use":args.bil_clus_vect_ind_to_use,\
                          "multi_feat_by_100":args.multi_feat_by_100,\
                          "cluster_lay_ind":args.bil_cluster_lay_ind,\
                          "clu_glob_vec":args.bil_clu_glob_vec,\
                          "clu_glob_rep_vec":args.bil_clu_glob_rep_vec,\
                          "clu_glob_corr_vec":args.bil_clu_glob_corr_vec,\
                          "clus_glob_norefine":args.bil_cluster_glob_norefine,\
                          "applySoftmaxOnSim_glob":args.apply_softmax_on_sim_glob,\
                          "clu_deconv":args.bil_clu_deconv}
                nbFeatAux = nbFeat
                if not args.bil_cluster_ensemble:

                    if args.bil_cluster_lay_ind != 4:
                        if nbFeat == 2048:
                            nbFeat = nbFeat//2**(4-args.bil_cluster_lay_ind)
                        elif nbFeat == 512:
                            nbFeat = nbFeat//2**(4-args.bil_cluster_lay_ind)
                        else:
                            raise ValueError("Unknown feature nb.")

                    if args.bil_clus_vect_ind_to_use == "all":
                        nbFeat *= args.resnet_bil_nb_parts
                    else:
                        nbFeat *= len(args.bil_clus_vect_ind_to_use.split(","))

                    if args.bil_clu_glob_vec:
                        nbFeat += getResnetFeat(args.first_mod, args.resnet_chan,args.deeplabv3_outchan)
                    elif args.bil_clu_glob_rep_vec or args.bil_clu_glob_corr_vec:
                        nbFeat += args.resnet_bil_nb_parts*getResnetFeat(args.first_mod, args.resnet_chan,args.deeplabv3_outchan)
            else:
                CNNconst = CNN2D
                kwargs = {"aux_model":args.aux_model,"bil_cluster_early":args.bil_cluster_early,"nb_parts":args.resnet_bil_nb_parts}
                nbFeatAux = nbFeat

            if args.first_mod.find("bagnet") == -1 and args.first_mod.find("hrnet") == -1:
                firstModel = CNNconst(args.first_mod, args.pretrained_visual, featMap=True,chan=args.resnet_chan, stride=args.resnet_stride,
                                      dilation=args.resnet_dilation, \
                                      num_classes=args.class_nb, \
                                      layerSizeReduce=args.resnet_layer_size_reduce,
                                      preLayerSizeReduce=args.resnet_prelay_size_reduce, \
                                      reluOnLast=args.relu_on_last_layer,
                                      deeplabv3_outchan=args.deeplabv3_outchan,\
                                      strideLay2=args.stride_lay2,strideLay3=args.stride_lay3,\
                                      strideLay4=args.stride_lay4,\
                                      multiple_stride=args.multiple_stride,\
                                      zoom_on_act=args.zoom_on_act,\
                                      dilOnStart=args.dil_on_start,\
                                      **kwargs)
            else:
                firstModel = CNNconst(args.first_mod, args.pretrained_visual,
                                        num_classes=args.class_nb,layerSizeReduce=args.resnet_layer_size_reduce,
                                        **kwargs)
        else:
            raise ValueError("Unknown visual model type : ", args.first_mod)

        if args.freeze_visual:
            for param in firstModel.parameters():
                param.requires_grad = False

    else:
        firstModel=None
        nbFeat = np.load("../results/{}_reprVec.npy".format(args.dataset_train.split("/")[-1])).shape[-1]
    ############### Second Model #######################

    zoomArgs= {"zoom":args.zoom,"zoom_max_sub_clouds":args.zoom_max_sub_clouds}

    if args.zoom and args.second_mod != "linear":
        raise ValueError("zoom must be used with linear second model")

    if args.second_mod == "linear":
        secondModel = LinearSecondModel(nbFeat,nbFeatAux, args.class_nb, args.dropout,args.aux_model,bil_cluster_ensemble=args.bil_cluster_ensemble,\
                                        bias=args.lin_lay_bias,aux_on_masked=args.aux_on_masked,**zoomArgs)
    else:
        raise ValueError("Unknown second model type : ", args.second_mod)

    ############### Whole Model ##########################

    net = Model(firstModel, secondModel,drop_and_crop=args.drop_and_crop,zoom=args.zoom,zoom_max_sub_clouds=args.zoom_max_sub_clouds,\
                zoom_merge_preds=args.zoom_merge_preds,reducedImgSize=args.reduced_img_size,upscaledTest=args.upscale_test)

    if args.cuda:
        net.cuda()

    if args.multi_gpu:
        net = DataParallelModel(net)

    return net

def addArgs(argreader):
    argreader.parser.add_argument('--first_mod', type=str, metavar='MOD',
                                  help='the net to use to produce feature for each frame')

    argreader.parser.add_argument('--dropout', type=float, metavar='D',
                                  help='The dropout amount on each layer of the RNN except the last one')

    argreader.parser.add_argument('--repr_vec_in_drop', type=float, metavar='S',
                                  help='The percentage of representative vectors dropped during training.')

    argreader.parser.add_argument('--second_mod', type=str, metavar='MOD',
                                  help='The temporal model. Can be "linear", "lstm" or "score_conv".')

    argreader.parser.add_argument('--freeze_visual', type=args.str2bool, metavar='BOOL',
                                  help='To freeze the weights of the visual model during training.')

    argreader.parser.add_argument('--pretrained_visual', type=args.str2bool, metavar='BOOL',
                                  help='To have a visual feature extractor pretrained on ImageNet.')

    argreader.parser.add_argument('--zoom', type=args.str2bool, metavar='BOOL',
                                  help='To use with a model that generates points. To zoom on the parts of the images where the points are focused an apply the model a second time on it.')

    argreader.parser.add_argument('--zoom_max_sub_clouds', type=int, metavar='NB',
                                  help='The maximum number of subclouds to use.')

    argreader.parser.add_argument('--zoom_merge_preds', type=args.str2bool, metavar='BOOL',
                                  help='To merge the predictions produced by the first model and by the model using crops.')

    argreader.parser.add_argument('--zoom_tied_models', type=args.str2bool, metavar='BOOL',
                                  help='To tie the weights of the global and the zoom model.')

    argreader.parser.add_argument('--zoom_model_no_topk', type=args.str2bool, metavar='BOOL',
                                  help='To force the zoom model to not use only the top-K pixels but all of them when the global model is a top-K model.')

    argreader.parser.add_argument('--aux_model', type=args.str2bool, metavar='INT',
                                  help='To train an auxilliary model that will apply average pooling and a dense layer on the feature map\
                        to make a prediction alongside the principal model\'s one.')

    argreader.parser.add_argument('--resnet_chan', type=int, metavar='INT',
                                  help='The channel number for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_stride', type=int, metavar='INT',
                                  help='The stride for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_dilation', type=int, metavar='INT',
                                  help='The dilation for the visual model when resnet is used')


    argreader.parser.add_argument('--resnet_layer_size_reduce', type=args.str2bool, metavar='INT',
                                  help='To apply a stride of 2 in the layer 2,3 and 4 when the resnet model is used.')
    argreader.parser.add_argument('--resnet_prelay_size_reduce', type=args.str2bool, metavar='INT',
                                  help='To apply a stride of 2 in the convolution and the maxpooling before the layer 1.')

    argreader.parser.add_argument('--stride_lay2', type=int, metavar='NB',
                                  help='Stride for layer 2.')
    argreader.parser.add_argument('--stride_lay3', type=int, metavar='NB',
                                  help='Stride for layer 3.')
    argreader.parser.add_argument('--stride_lay4', type=int, metavar='NB',
                                  help='Stride for layer 4.')

    argreader.parser.add_argument('--reduced_img_size', type=int, metavar='BOOL',
                                  help="The size at which the image is reduced at the begining of the process")

    argreader.parser.add_argument('--norm_points', type=args.str2bool, metavar='BOOL',
                                  help="To normalize the points before passing them to pointnet")

    argreader.parser.add_argument('--relu_on_last_layer', type=args.str2bool, metavar='BOOL',
                                  help="To apply relu on the last layer of the feature extractor.")

    argreader.parser.add_argument('--deeplabv3_outchan', type=int, metavar='BOOL',
                                  help="The number of output channel of deeplabv3")

    argreader.parser.add_argument('--resnet_bil_nb_parts', type=int, metavar='INT',
                                  help="The number of parts for the bilinear model.")
    argreader.parser.add_argument('--resnet_bilinear', type=args.str2bool, metavar='BOOL',
                                  help="To use bilinear attention")
    argreader.parser.add_argument('--bil_center_loss', type=args.str2bool, metavar='BOOL',
                                  help="To use center loss when using bilinear model")
    argreader.parser.add_argument('--bil_center_loss_beta', type=float, metavar='BOOL',
                                  help="The update rate term for the center loss.")

    argreader.parser.add_argument('--bil_cluster', type=args.str2bool, metavar='BOOL',
                                  help="To have a cluster bilinear")
    argreader.parser.add_argument('--bil_cluster_ensemble', type=args.str2bool, metavar='BOOL',
                                  help="To classify each of the feature vector obtained and then aggregates those decision.")
    argreader.parser.add_argument('--apply_softmax_on_sim', type=args.str2bool, metavar='BOOL',
                                  help="Apply softmax on similarity computed during clustering.")
    argreader.parser.add_argument('--softm_coeff', type=float, metavar='BOOL',
                                  help="The softmax temperature. The higher it is, the sharper weights will be.")
    argreader.parser.add_argument('--bil_clust_unnorm', type=args.str2bool, metavar='BOOL',
                                  help="To mulitply similarity by norm to make weights superior to 1.")
    argreader.parser.add_argument('--bil_clus_vect_gate', type=args.str2bool, metavar='BOOL',
                                  help="To add a gate that reorder the vectors.")

    argreader.parser.add_argument('--bil_clus_vect_ind_to_use',type=str, metavar='BOOL',
                                  help="Specify this list to only use some of the vectors collected. Eg : --bil_clus_vect_ind_to_use 1,2")

    argreader.parser.add_argument('--bil_clust_update_sco_by_norm_sim', type=args.str2bool, metavar='BOOL',
                                  help="To update score using normalised similarity.")

    argreader.parser.add_argument('--bil_norm_feat', type=args.str2bool, metavar='BOOL',
                                  help="To normalize feature before computing attention")

    argreader.parser.add_argument('--drop_and_crop', type=args.str2bool, metavar='BOOL',
                                  help="To crop and drop part of the images where the attention is focused.")

    argreader.parser.add_argument('--hid_lay', type=args.str2bool, metavar='BOOL',
                                  help="To add a hiddent layer before the softmax layer")

    argreader.parser.add_argument('--bil_cluster_ensemble_gate', type=args.str2bool, metavar='BOOL',
                                  help="To add a gate network at the end of the cluster ensemble network.")
    argreader.parser.add_argument('--bil_cluster_ensemble_gate_drop', type=args.str2bool, metavar='BOOL',
                                  help="To drop the feature vector with the most important weight.")
    argreader.parser.add_argument('--bil_cluster_ensemble_gate_randdrop', type=args.str2bool, metavar='BOOL',
                                  help="To randomly drop one feature vector.")

    argreader.parser.add_argument('--bil_cluster_norefine', type=args.str2bool, metavar='BOOL',
                                  help="To not refine feature vectors by using similar vectors.")
    argreader.parser.add_argument('--bil_cluster_randvec', type=args.str2bool, metavar='BOOL',
                                  help="To select random vectors as initial estimation instead of vectors with high norms.")
    argreader.parser.add_argument('--bil_cluster_glob_norefine', type=args.str2bool, metavar='BOOL',
                                  help="To prevent refining when extracting representative vectors at the last layer.")

    argreader.parser.add_argument('--bil_cluster_lay_ind', type=int, metavar='BOOL',
                                  help="The layer at which to group pixels.")

    argreader.parser.add_argument('--bil_cluster_early', type=args.str2bool, metavar='BOOL',
                                  help="To perform early grouping.")
    argreader.parser.add_argument('--bil_clu_earl_exp', type=args.str2bool, metavar='BOOL',
                                  help="To apply soft-max when using early grouping.")

    argreader.parser.add_argument('--bil_clu_glob_vec', type=args.str2bool, metavar='BOOL',
                                  help="To compute a global vector by global average pooling on the last layer.")
    argreader.parser.add_argument('--bil_clu_glob_rep_vec', type=args.str2bool, metavar='BOOL',
                                  help="To compute representative vectors on the last layer.")
    argreader.parser.add_argument('--bil_clu_glob_corr_vec', type=args.str2bool, metavar='BOOL',
                                  help="To extract the vector on the last layer at the position where \
                                            representative vectors have been extracted.")
    argreader.parser.add_argument('--apply_softmax_on_sim_glob', type=args.str2bool, metavar='BOOL',
                              help="When extracting representative vectors at the last layer, whether to apply softmax.")
    argreader.parser.add_argument('--bil_clu_deconv', type=args.str2bool, metavar='BOOL',
                              help="To apply deconvolution to recover high resolution attention maps.")

    argreader.parser.add_argument('--lin_lay_bias', type=args.str2bool, metavar='BOOL',
                                  help="To add a bias to the final layer.")

    argreader.parser.add_argument('--multi_feat_by_100', type=args.str2bool, metavar='BOOL',
                                  help="To multiply feature by 100 when using bilinear model.")

    argreader.parser.add_argument('--aux_on_masked', type=args.str2bool, metavar='BOOL',
                                  help="To train dense layers on masked version of the feature matrix.")

    argreader.parser.add_argument('--multiple_stride', type=args.str2bool, metavar='BOOL',
                                  help="To compute multiple stride during test to obtain a detailed final feature map.")

    argreader.parser.add_argument('--zoom_on_act', type=args.str2bool, metavar='BOOL',
                                  help="To zoom on feature map during training.")

    argreader.parser.add_argument('--dil_on_start', type=args.str2bool, metavar='BOOL',
                                  help="To apply dilation at the begining of blocks")

    argreader.parser.add_argument('--master_net', type=args.str2bool, help='To distill a master network into the trained network.')
    argreader.parser.add_argument('--m_net_path', type=str, help='The path to the master network')
    argreader.parser.add_argument('--m_conf_path', type=str, help='The path to the master network config file.')
    argreader.parser.add_argument('--kl_interp', type=float, help='If set to 0, will use regular target, if set to 1, will only use master net target')
    argreader.parser.add_argument('--kl_temp', type=float, help='KL temperature.')

    return argreader
