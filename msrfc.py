import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F
from math import sqrt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num
def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size[0] - kernel_size + (2 * padding)) / stride) + 1),int(((image_size[1] - kernel_size + (2 * padding)) / stride) + 1)



class VGGnet(nn.Module):

    def __init__(self, input_channel):
        super().__init__()

        layers=[64,128,256,512]

        self.conv1 = self._conv(input_channel,layers[0])
        self.maxp1 = nn.MaxPool2d(2,stride=2)
        self.conv2 = self._conv(layers[0],layers[1])
        self.maxp2 = nn.MaxPool2d(2,stride=2)
        self.conv3 = self._conv(layers[1],layers[2])
        self.maxp3 = nn.MaxPool2d(2,stride=2)
        self.conv4 = self._conv(layers[2],layers[3])
        self.maxp4 = nn.MaxPool2d(2,stride=2)


    def _conv(self,inplance,outplance,nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance,outplance,kernel_size=3,
                          stride=1,padding=1,bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.ReLU(inplace=True))
            inplance = outplance

        conv = nn.Sequential(*conv)

        return conv

    def forward(self, x):
        xlist=[x]
        x = self.conv1(x)
        xlist.append(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        xlist.append(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        xlist.append(x)
        x = self.maxp3(x)
        x = self.conv4(x)
        xlist.append(x)
        return xlist
class VGGnet(nn.Module):

    def __init__(self, input_channel):
        super().__init__()

        layers=[64,128,256,512]

        self.conv1 = self._conv(input_channel,layers[0])
        self.maxp1 = nn.MaxPool2d(2,stride=2)
        self.conv2 = self._conv(layers[0],layers[1])
        self.maxp2 = nn.MaxPool2d(2,stride=2)
        self.conv3 = self._conv(layers[1],layers[2])
        self.maxp3 = nn.MaxPool2d(2,stride=2)
        self.conv4 = self._conv(layers[2],layers[3])
        self.maxp4 = nn.MaxPool2d(2,stride=2)


    def _conv(self,inplance,outplance,nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance,outplance,kernel_size=3,
                          stride=1,padding=1,bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.ReLU(inplace=True))
            inplance = outplance

        conv = nn.Sequential(*conv)

        return conv

    def forward(self, x):
        xlist=[x]
        x = self.conv1(x)
        xlist.append(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        xlist.append(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        xlist.append(x)
        x = self.maxp3(x)
        x = self.conv4(x)
        xlist.append(x)
        return xlist
class exchange(nn.Module):
    def __init__(self,scale1,scale2,k_1,k_2):
        super().__init__()
        self.layers1 = [scale1,scale1+k_1+k_1,scale1+2*k_1+k_1,scale1+3*k_1+k_1,scale1+4*k_1+k_1]
        self.layers2 = [scale2,scale2+k_2+k_1,scale2+2*k_2+k_1,scale2+3*k_2+k_1,scale2+4*k_2+k_1]
        self.x1 = nn.Sequential(nn.Conv2d(scale1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y1 = nn.Sequential(nn.Conv2d(scale2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x1c = nn.Sequential(nn.Conv2d(scale1,k_1,kernel_size=4,stride=2,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y1t = nn.Sequential(nn.ConvTranspose2d(scale2,k_1,kernel_size=4,stride=2,padding=1,bias=False),nn.ReLU(inplace=True))
        
        self.x2_input = nn.Sequential(nn.Conv2d(self.layers1[1],k_1,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        self.y2_input = nn.Sequential(nn.Conv2d(self.layers2[1],k_2,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        
        
        self.x2 = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y2 = nn.Sequential(nn.Conv2d(k_2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x2c = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=4,stride=2,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y2t = nn.Sequential(nn.ConvTranspose2d(k_2,k_1,kernel_size=4,stride=2,padding=1,bias=False),nn.ReLU(inplace=True))
        
        self.x3_input = nn.Sequential(nn.Conv2d(self.layers1[2],k_1,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        self.y3_input = nn.Sequential(nn.Conv2d(self.layers2[2],k_2,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        
        
        self.x3 = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y3 = nn.Sequential(nn.Conv2d(k_2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x3c = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=4,stride=2,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y3t = nn.Sequential(nn.ConvTranspose2d(k_2,k_1,kernel_size=4,stride=2,padding=1,bias=False),nn.ReLU(inplace=True))
        
        self.x4_input = nn.Sequential(nn.Conv2d(self.layers1[3],k_1,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        self.y4_input = nn.Sequential(nn.Conv2d(self.layers2[3],k_2,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        
        self.x4 = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y4 = nn.Sequential(nn.Conv2d(k_2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x4c = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=4,stride=2,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y4t = nn.Sequential(nn.ConvTranspose2d(k_2,k_1,kernel_size=4,stride=2,padding=1,bias=False),nn.ReLU(inplace=True))
        
        self.x5_input = nn.Sequential(nn.Conv2d(self.layers1[4],k_1,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        self.y5_input = nn.Sequential(nn.Conv2d(self.layers2[4],k_2,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        
        self.x5 = nn.Sequential(nn.Conv2d(self.layers1[4],scale1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y5 = nn.Sequential(nn.Conv2d(self.layers2[4],scale2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        
        
        
        
    def _forward(self,ten1,ten2):
        x1 = self.x1(ten1)
        y1 = self.y1(ten2)
        x1c = self.x1c(ten1)
        y1t = self.y1t(ten2)
        
        x2_input = torch.cat([ten1,x1,y1t],1)
        y2_input = torch.cat([ten2,y1,x1c],1)
        x2_input = self.x2_input(x2_input)
        y2_input = self.y2_input(y2_input)
        
        x2 = self.x2(x2_input)
        y2 = self.y2(y2_input)
        x2c = self.x2c(x2_input)
        y2t = self.y2t(y2_input)
        
        x3_input = torch.cat([ten1,x1,x2,y2t],1)
        y3_input = torch.cat([ten2,y1,y2,x2c],1)
        x3_input = self.x3_input(x3_input)
        y3_input = self.y3_input(y3_input)
        
        x3 = self.x3(x3_input)
        y3 = self.y3(y3_input)
        x3c = self.x3c(x3_input)
        y3t = self.y3t(y3_input)
        
        x4_input = torch.cat([ten1,x1,x2,x3,y3t],1)
        y4_input = torch.cat([ten2,y1,y2,y3,x3c],1)
        x4_input = self.x4_input(x4_input)
        y4_input = self.y4_input(y4_input)
        
        x4 = self.x4(x4_input)
        y4 = self.y4(y4_input)
        x4c = self.x4c(x4_input)
        y4t = self.y4t(y4_input)
        
        x5_input = torch.cat([ten1,x1,x2,x3,x4,y4t],1)
        y5_input = torch.cat([ten2,y1,y2,y3,y4,x4c],1)
        #x5_input = self.x5_input(x5_input)
        #y5_input = self.y5_input(y5_input)
        x5 = self.x5(x5_input)
        y5 = self.y5(y5_input)
        
        return 0.4*x5+ten1, 0.4*y5+ten2
        
        
        
        
        
    def forward(self,ten):
        return self._forward(ten[0],ten[1])


class MSF(nn.Module):
    def __init__(self,inplace,num_classes):
        super().__init__()
        self.net = VGGnet(inplace)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512,num_classes)
        #self.exchange12_one = exchange(64,3216,32)
        #self.exchange34_one = exchange(128,256,32,64)
        
        #self.exchange23_one = exchange(64,128,32,32)
        
        #self.exchange12_two = exchange(32,64,16,32)
        #self.exchange34_two = exchange(128,256,32,64)
        self.exchange12_one = exchange(64,128,16,32)
        self.att1_one = SpatialAttention(64,64)
        self.att2_one = SpatialAttention(128,128)
        self.att3_one = SpatialAttention(256,256)
        self.att4_one = SpatialAttention(512,512)
        self.exchange34_one = exchange(256,512,32,64)
        self.exchange23_one = exchange(128,256,32,32)

        self.exchange12_two = exchange(64,128,16,32)
        self.exchange34_two = exchange(256,512,32,128)

    def forward(self,x):
        xlist = self.net(x)
        n11,n12,n13,n14 = xlist[1],xlist[2],xlist[3],xlist[4]
        #print(n11.shape,n12.shape,n13.shape,n14.shape)
        n11 = self.att1_one(n11)
        n12 = self.att2_one(n12)
        n13 = self.att3_one(n13)
        n14 = self.att4_one(n14)
        n21,n22 = self.exchange12_one((n11,n12))
        n23,n24 = self.exchange34_one((n13,n14))
        n22,n23 = self.exchange23_one((n22,n23))
        
        n31,n32 = self.exchange12_two((n21,n22))
        n33,n34 = self.exchange34_two((n23,n24))
        li = [n14,n24,n34]
        logits_list = []
        for l in li:
            r = torch.flatten(self.avg(l),1)
            c = self.classifier(r)
            logits_list.append(c)
        combined_logits = 0
        for r in logits_list:
            combined_logits += r
        return logits_list,combined_logits

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

def _conv_spa(inplance,outplance,nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance,outplance,kernel_size=3,
                          stride=1,padding=1,bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.ReLU(inplace=True))
            inplance = outplance

        conv = nn.Sequential(*conv)

        return conv

class SpatialAttention(nn.Module):
    def __init__(self,in_channel,out_channel, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = _conv_spa(in_channel,out_channel)
        self.conv2 = nn.Conv2d(out_channel, 1, kernel_size=1,bias=False)
        
        self.conv_main = _conv_spa(in_channel,out_channel)
        self.sigmoid = nn.Sigmoid()
        self.sande = SE_Block(out_channel,16)
        self.convcombine = nn.Conv2d(2*out_channel,out_channel,kernel_size=1,stride=1,padding=1,bias=False)
        self.bncombine = nn.BatchNorm2d(out_channel)
        self.relucombine = nn.ReLU(inplace=True)
    def forward(self, x):
        sp = self.conv1(x)
        sp = self.conv2(sp)
        act = self.sigmoid(sp)
        x_spatial = x*act
        #x_se = self.sande(x)
        #x = torch.cat([x_spatial, x_se], dim=1)
        #x = self.convcombine(x)
        x = self.bncombine(x_spatial)
        return self.relucombine(x)



class VGGnet_spatial(nn.Module):

    def __init__(self, input_channel,num_classes):
        super().__init__()

        layers=[64,128,256,512]

        self.conv1 = self._conv(input_channel,layers[0])
        self.att1 = SpatialAttention(layers[0],layers[0])
        self.maxp1 = nn.MaxPool2d(2,stride=2)

        self.conv2 = self._conv(layers[0],layers[1])
        self.att2 = SpatialAttention(layers[1],layers[1])
        self.maxp2 = nn.MaxPool2d(2,stride=2)

        self.conv3 = self._conv(layers[1],layers[2])
        self.att3 = SpatialAttention(layers[2],layers[2])
        self.maxp3 = nn.MaxPool2d(2,stride=2)

        self.conv4 = self._conv(layers[2],layers[3])
        self.att4 = SpatialAttention(layers[3],layers[3])
        self.maxp4 = nn.MaxPool2d(2,stride=2)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512,num_classes)
    
    
    def _conv(self,inplance,outplance,nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance,outplance,kernel_size=3,
                          stride=1,padding=1,bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.ReLU(inplace=True))
            inplance = outplance

        conv = nn.Sequential(*conv)

        return conv

    def forward(self, x):
        xlist=[x]
        x = self.conv1(x)
        x = self.att1(x)
        x = self.maxp1(x)
        
        x = self.conv2(x)
        x = self.att2(x)
        x = self.maxp2(x)
        
        x = self.conv3(x)
        x = self.att3(x)
        x = self.maxp3(x)
        
        x = self.conv4(x)
        x = self.att4(x)
        r = torch.flatten(self.avg(x),1)
        c = self.classifier(r)
        return c

img = torch.randn(1, 1, 64,128)
v = VGGnet_spatial(input_channel=1,num_classes=657)
preds = v(img) # (1, 1000)
print(preds.shape)
