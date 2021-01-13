import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import numpy as np
from  torch.nn.modules.upsampling import Upsample

from torch.nn.functional import interpolate as interpo
import sys
import glob
import torch

from skimage.transform import resize
import torch.nn.functional as F

import modelBuilder
import torchvision
'''

Just a modification of the torchvision resnet model to get the before-to-last activation


'''

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1,dilation=1,groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False,dilation=dilation,groups=groups,padding_mode="circular")


def conv1x1(in_planes, out_planes, stride=1,groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)

def conv3x3Transp(in_planes, out_planes, stride=1,dilation=1,groups=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(out_planes,in_planes, kernel_size=3, stride=stride,\
                                padding=dilation,bias=False,dilation=dilation,groups=groups)

def conv1x1Transp(in_planes, out_planes, stride=1,groups=1):
    """1x1 convolution"""
    return nn.ConvTranspose2d(out_planes,in_planes, kernel_size=1, stride=stride, bias=False,groups=groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,endRelu=True,dilation=1,groups=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,dilation,groups=groups)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes,groups=groups)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.endRelu = endRelu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        if self.endRelu:
            out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,endRelu=False,dilation=1,\
                            multiple_stride=False):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride,dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.endRelu = endRelu
        self.multiple_stride = multiple_stride

    def applyLastConvs(self,x,identity):
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identityDown = self.downsample(identity)
            out += identityDown
        else:
            out += identity

        if self.endRelu:
            out = self.relu(out)

        return out

    def forward(self, inp):

        mult_str = self.stride > 1 and self.multiple_stride

        retDic = {}
        for key in inp.keys():

            x = inp[key]

            identity = x

            inChan = x.size(1)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            if mult_str:

                for i in range(self.stride):
                    for j in range(self.stride):
                        newKey = "{}_{}{}".format(key,i,j) if len(inp) > 1 else "{}{}".format(i,j)
                        retDic[newKey] = self.applyLastConvs(out[:,:,i:,j:],identity)
            else:
                out = self.applyLastConvs(out,identity)
                retDic[key] = out

        return retDic

class RevBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,endRelu=False,dilation=1):
        super(RevBottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1Transp(inplanes, planes)
        self.conv2 = conv3x3Transp(planes, planes, stride,dilation)
        self.conv3 = conv1x1Transp(planes, planes * self.expansion)
        self.stride = stride

    def forward(self, x):
        out = self.conv3(x)
        out = self.conv2(out)
        out = self.conv1(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None,maxPoolKer=(3,3),maxPoolPad=(1,1),stride=(2,2),\
                    strideLay2=2,strideLay3=2,strideLay4=2,\
                    featMap=False,chan=64,inChan=3,dilation=1,layerSizeReduce=True,preLayerSizeReduce=True,layersNb=4,reluOnLast=False,\
                    bil_cluster_early=False,nb_parts=3,bil_clu_earl_exp=False,multiple_stride=False,bin_multiple_stride=True,\
                    zoom_on_act=False,dilOnStart=False):

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if not type(chan) is list:
            chan = [chan,chan*2,chan*4,chan*8]

        self.inplanes = chan[0]

        self.conv1 = nn.Conv2d(inChan, chan[0], kernel_size=7, stride=1 if not preLayerSizeReduce else stride,bias=False,padding=3)
        self.bn1 = norm_layer(chan[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=maxPoolKer, stride=1 if not preLayerSizeReduce else stride, padding=maxPoolPad)

        if type(dilation) is int:
            dilation = [dilation,dilation,dilation]
        elif len(dilation) != 3:
            raise ValueError("dilation must be a list of 3 int or an int.")

        self.nbLayers = len(layers)

        self.binMultStr = bin_multiple_stride
        self.dilOnStart = dilOnStart

        #All layers are built but they will not necessarily be used
        self.layer1 = self._make_layer(block, chan[0], layers[0], stride=1,norm_layer=norm_layer,reluOnLast=reluOnLast if self.nbLayers==1 else True,\
                                        dilation=1)
        self.layer2 = self._make_layer(block, chan[1], layers[1], stride=1 if not layerSizeReduce else strideLay2, norm_layer=norm_layer,\
                                        reluOnLast=reluOnLast if self.nbLayers==2 else True,dilation=dilation[0])
        self.layer3 = self._make_layer(block, chan[2], layers[2], stride=1 if not layerSizeReduce else strideLay3, norm_layer=norm_layer,\
                                        reluOnLast=reluOnLast if self.nbLayers==3 else True,dilation=dilation[1],multStr=multiple_stride if not self.binMultStr else False)
        self.layer4 = self._make_layer(block, chan[3], layers[3], stride=1 if not layerSizeReduce else strideLay4, norm_layer=norm_layer,\
                                        reluOnLast=reluOnLast if self.nbLayers==4 else True,dilation=dilation[2],multStr=multiple_stride)

        if layersNb<1 or layersNb>4:
            raise ValueError("Wrong number of layer : ",layersNb)

        self.layersNb = layersNb
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(chan[0]*(2**(4-1)) * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.featMap = featMap

        self.bil_cluster_early = bil_cluster_early
        if self.bil_cluster_early:
            self.clus_earl_conv1x1 = conv1x1(chan[3],chan[3]//nb_parts)
            self.nb_parts = nb_parts
            self.bil_clu_earl_exp = bil_clu_earl_exp

        self.multiple_stride = multiple_stride
        if self.multiple_stride:
            self.rowInds = None
            self.colInds = None
            self.rows = None
            self.cols = None
            if strideLay2 != 2 or strideLay3 != 2 or strideLay4 != 2:
                raise ValueError("Multiple stride not implemented for stride != 2")

        self.zoom_on_act = zoom_on_act
    def createMultStrInds(self,x,bin=False):

        if not bin:
            self.rowInds = torch.arange(x.size(2)).to(x.device)
            self.colInds = torch.arange(x.size(3)).to(x.device)

            self.rows = {rem:(self.rowInds % 4 == rem) for rem in range(4)}
            self.cols = {rem:(self.colInds % 4 == rem) for rem in range(4)}
        else:
            self.rowInds = torch.arange(x.size(2)).to(x.device)
            self.colInds = torch.arange(x.size(3)).to(x.device)

            self.rows = {rem:(self.rowInds % 2 == rem) for rem in range(2)}
            self.cols = {rem:(self.colInds % 2 == rem) for rem in range(2)}

    def fillMap(self,map,values,cond1,cond2,batchSize,chanNb):
        mask = self.binary(cond1,cond2,batchSize,chanNb)
        map[mask] = values.reshape(-1)
        return map

    def binary(self,cond1,cond2,batchSize,chanNb):
        binaryArr = (cond1).unsqueeze(1)*(cond2).unsqueeze(0)
        binaryArr = binaryArr.unsqueeze(0).unsqueeze(0).expand(batchSize,chanNb,-1,-1)
        return binaryArr

    def gatherMultStr(self,x,bin=False):
        if not bin:
            feat = torch.zeros(x["00_00"].size(0),x["00_00"].size(1),len(self.rowInds),len(self.colInds)).to(x["00_00"].device)
            for key in x:
                row = int(key.split("_")[1][0])*2+int(key.split("_")[0][0])
                col = int(key.split("_")[1][1])*2+int(key.split("_")[0][1])
                feat = self.fillMap(feat,x[key],self.rows[row],self.cols[col],x["00_00"].size(0),x["00_00"].size(1))
        else:
            feat = torch.zeros(x["00"].size(0),x["00"].size(1),len(self.rowInds),len(self.colInds)).to(x["00"].device)
            for key in x:
                row = int(key[0])
                col = int(key[1])
                feat = self.fillMap(feat,x[key],self.rows[row],self.cols[col],x["00"].size(0),x["00"].size(1))

        return feat

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None,reluOnLast=False,dilation=1,multStr=False):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        if self.dilOnStart:
            layers.append(block(self.inplanes, planes, stride, downsample, norm_layer,multiple_stride=multStr,dilation=dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, norm_layer,multiple_stride=multStr))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            if self.dilOnStart:
                layers.append(block(self.inplanes, planes, norm_layer=norm_layer,endRelu=True))
            else:
                layers.append(block(self.inplanes, planes, norm_layer=norm_layer,endRelu=True,dilation=dilation))

        return nn.Sequential(*layers)

    def crop(self,x):
        mapAct = x.sum(dim=1)
        inds = torch.max(mapAct.view(mapAct.size(0),-1),dim=-1)[1]
        rows,cols = inds//x.size(3),inds%x.size(3)

        y1 = torch.clamp(rows-x.size(2)//4,0,x.size(2)//2)
        y2 = y1 + x.size(2)//2
        x1 = torch.clamp(cols-x.size(3)//4,0,x.size(3)//2)
        x2 = x1 + x.size(3)//2

        crop = []
        for i in range(len(x)):
            crop.append(x[i:i+1,:,y1[i]:y2[i],x1[i]:x2[i]])
        x = torch.cat(crop,dim=0)
        return x

    def forward(self,xInp,returnLayer="last"):

        retDict = {}
        x = self.conv1(xInp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1({"00":x})
        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        if self.training and self.zoom_on_act:
            x3["00"] = self.crop(x3["00"])

        x4 = self.layer4(x3)
        if self.training and self.zoom_on_act:
            x4["00"] = self.crop(x4["00"])

        if not self.featMap:
            x = self.avgpool(x4["00"])
            x = x.view(x.size(0), -1)
            retDict["x"] = x
        elif self.multiple_stride:

            if self.rowInds is None:
                self.createMultStrInds(x2["00"] if not self.binMultStr else x3["00"],bin=self.binMultStr)

            retDict["x"] = self.gatherMultStr(x4,bin=self.binMultStr)

        else:
            retDict["x"] = x4["00"]

        retDict["layerFeat"] = {1:x1[list(x1.keys())[0]],\
                                2:x2[list(x2.keys())[0]],\
                                3:x3[list(x3.keys())[0]],\
                                4:x4[list(x4.keys())[0]]}

        return retDict

class RevResNet(ResNet):

    def __init__(self, **kwargs):
        super(RevResNet, self).__init__(**kwargs)

    def forward(self,x,returnLayer="last"):

        retDict = {}

        for layer in [self.layer4, self.layer3]:
            for i in range(len(layer)-1,-1,-1):
                x = layer[i](x)

        retDict["x"] = x

        return retDict

def removeTopLayer(params):
    params.pop("fc.weight")
    params.pop("fc.bias")
    return params

def resnet4(pretrained=False,chan=8, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], chan=chan,layersNb=1,**kwargs)

    if pretrained and chan != 64:
        raise ValueError("ResNet4 with {} channel does not have pretrained weights on ImageNet.".format(chan))
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def resnet9_att(pretrained=False,chan=8,attChan=16,attBlockNb=1, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], chan=chan,layersNb=2,attention=True,attChan=attChan,attBlockNb=attBlockNb,**kwargs)

    if pretrained and chan != 64:
        raise ValueError("ResNet9 with {} channel does not have pretrained weights on ImageNet.".format(chan))
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)

        paramsToLoad = {}
        for key in params:
            if key in model.state_dict() and model.state_dict()[key].size() == params[key].size():
                paramsToLoad[key] = params[key]
        params = paramsToLoad

        model.load_state_dict(params,strict=False)
    return model


def resnet9(pretrained=False,chan=8, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], chan=chan,layersNb=2,**kwargs)

    if pretrained and chan != 64:
        raise ValueError("ResNet9 with {} channel does not have pretrained weights on ImageNet.".format(chan))
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def resnet14_att(pretrained=False,chan=8,attChan=16,attBlockNb=1, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], chan=chan,layersNb=3,attention=True,attChan=attChan,attBlockNb=attBlockNb,**kwargs)

    if pretrained and chan != 64:
        raise ValueError("ResNet14 with {} channel does not have pretrained weights on ImageNet.".format(chan))
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)

        paramsToLoad = {}
        for key in params:
            if key in model.state_dict() and model.state_dict()[key].size() == params[key].size():
                paramsToLoad[key] = params[key]
        params = paramsToLoad

        model.load_state_dict(params,strict=False)
    return model

def resnet18_att(pretrained=False, strict=True,attChan=16,attBlockNb=1,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],attention=True,attChan=attChan,attBlockNb=attBlockNb, **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet34'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model


def resnet50(pretrained=False, strict=True,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet50'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def revresnet50(pretrained=False,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs["block"] = RevBottleneck
    kwargs["layers"] = [3, 4, 6, 3]
    model = RevResNet(**kwargs)

    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet101'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet152'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model
