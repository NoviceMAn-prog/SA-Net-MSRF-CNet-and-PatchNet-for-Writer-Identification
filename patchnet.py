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
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class Pool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.downsample = DepthWiseConv2d(dim, dim * 2, kernel_size = 3, stride = 2, padding = 1)
        self.cls_ff = nn.Linear(dim, dim * 2)

    def forward(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]

        cls_token = self.cls_ff(cls_token)

        tokens = rearrange(tokens, 'b (h w) c -> b c h w', h = int(sqrt(tokens.shape[1])))
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, 'b c h w -> b (h w) c')

        return torch.cat((cls_token, tokens), dim = 1)
class PiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert isinstance(depth, tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'
        heads = cast_tuple(heads, len(depth))
        self.pool='mean'
        self.to_latent = nn.Identity()
        patch_dim = patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size = patch_size, stride = patch_size // 2),
            Rearrange('b c n -> b n c'),
            nn.Linear(patch_dim, dim)
        )

        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        layers = []
        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)

            layers.append(Transformer(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout))

            if not_last:
                layers.append(Pool(dim))
                dim *= 2

        self.layers = nn.Sequential(
            *layers,
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes))

    def forward(self, img):
        #print(img.shape)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for l in self.layers[0:len(self.layers)-1]:
            x = l(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.layers[-1](x)
        return x
v = PiT(
    image_size = 128,
    patch_size = 16,
    dim = 128,
    num_classes = 105,
    depth = (3, 3, 3),     # list of depths, indicating the number of rounds of each stage before a downsample
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)

# forward pass now returns predictions and the attention maps

img = torch.randn(1, 1, 128,128)

preds = v(img) # (1, 1000)
print(preds.shape)



class exchange(nn.Module):
    def __init__(self,scale1,scale2,k_1,k_2):
        super().__init__()
        self.layers1 = [scale1,scale1+k_1+k_1,scale1+2*k_1+k_1,scale1+3*k_1+k_1,scale1+4*k_1+k_1]
        self.layers2 = [scale2,scale2+k_2+k_1,scale2+2*k_2+k_1,scale2+3*k_2+k_1,scale2+4*k_2+k_1]
        self.x1 = nn.Sequential(nn.Conv2d(scale1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y1 = nn.Sequential(nn.Conv2d(scale2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x1c = nn.Sequential(nn.Conv2d(scale1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y1t = nn.Sequential(nn.ConvTranspose2d(scale2,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        
        self.x2_input = nn.Sequential(nn.Conv2d(self.layers1[1],k_1,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        self.y2_input = nn.Sequential(nn.Conv2d(self.layers2[1],k_2,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        
        
        self.x2 = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y2 = nn.Sequential(nn.Conv2d(k_2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x2c = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y2t = nn.Sequential(nn.ConvTranspose2d(k_2,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        
        self.x3_input = nn.Sequential(nn.Conv2d(self.layers1[2],k_1,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        self.y3_input = nn.Sequential(nn.Conv2d(self.layers2[2],k_2,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        
        
        self.x3 = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y3 = nn.Sequential(nn.Conv2d(k_2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x3c = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y3t = nn.Sequential(nn.ConvTranspose2d(k_2,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        
        self.x4_input = nn.Sequential(nn.Conv2d(self.layers1[3],k_1,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        self.y4_input = nn.Sequential(nn.Conv2d(self.layers2[3],k_2,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        
        self.x4 = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y4 = nn.Sequential(nn.Conv2d(k_2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x4c = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y4t = nn.Sequential(nn.ConvTranspose2d(k_2,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        
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
        return x

class dfrag_old(nn.Module):
    def __init__(self,inplace,num_classes):
        super().__init__()
        self.net = VGGnet(inplace)
        self.avg = nn.AdaptiveAvgPool2d(1)
        
        self.conv1_stream1 = self._conv(1,64)
        self.conv1_stream2 = self._conv(1,64)
        self.conv1_stream3 = self._conv(1,64)
        self.conv1_stream4 = self._conv(1,64)
        self.conv1_stream5 = self._conv(1,64)
        self.maxp1 = nn.MaxPool2d(2,stride=(2,2))
        
        self.conv2_stream1 = self._conv(64,128)
        self.conv2_stream2 = self._conv(64,128)
        self.conv2_stream3 = self._conv(64,128)
        self.conv2_stream4 = self._conv(64,128)
        self.conv2_stream5 = self._conv(64,128)
        self.maxp2 = nn.MaxPool2d(2,stride=2)
        
        self.classifier = nn.Linear(512,num_classes)
        self.exchange12_one = exchange(64,64,16,16)
        self.exchange23_one = exchange(64,64,16,16)
        self.exchange34_one = exchange(64,64,16,16)
        self.exchange45_one = exchange(64,64,16,16)
        
        self.exchange12_2 = exchange(128,128,32,32)
        self.exchange23_2 = exchange(128,128,32,32)
        self.exchange34_2 = exchange(128,128,32,32)
        self.exchange45_2 = exchange(128,128,32,32)
        self.classifier_global = nn.Linear(512,num_classes)
        self.classifier_patch = nn.Linear(128,num_classes)
        
    
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
    
    def forward(self,x):
        x_global = self.net(x)
        step = 16
        
        #print(xlist[0].shape)
        xpatch = []
        # input image
        reslist = []
        for n in range(0,65,step):
            xpatch.append(x[:,:,:,n:n+64])
            
        x1 = self.conv1_stream1(xpatch[0])
        x1 = self.maxp1(x1)
        x2 = self.conv1_stream2(xpatch[1])
        x2 = self.maxp1(x2)
        x3 = self.conv1_stream3(xpatch[2])
        x3= self.maxp1(x3)
        x4 = self.conv1_stream4(xpatch[3])
        x4 = self.maxp1(x4)
        x5 = self.conv1_stream5(xpatch[4])
        x5 = self.maxp1(x5)
        
        x1,x2 = self.exchange12_one((x1,x2))
        x2,x3 = self.exchange23_one((x2,x3))
        x3,x4 = self.exchange34_one((x3,x4))
        x4,x5 = self.exchange45_one((x4,x5))
        
        x1 = self.conv2_stream1(x1)
        x1 = self.maxp2(x1)
        x2 = self.conv2_stream2(x2)
        x2 = self.maxp2(x2)
        x3 = self.conv2_stream3(x3)
        x3= self.maxp2(x3)
        x4 = self.conv2_stream4(x4)
        x4 = self.maxp2(x4)
        x5 = self.conv2_stream5(x5)
        x5 = self.maxp2(x5)
        
        x1,x2 = self.exchange12_2((x1,x2))
        x2,x3 = self.exchange23_2((x2,x3))
        x3,x4 = self.exchange34_2((x3,x4))
        x4,x5 = self.exchange45_2((x4,x5))
        
        
        li = [x1,x2,x3,x4,x5]
        global_pred = torch.flatten(self.avg(x_global),1)
        global_logs = self.classifier_global(global_pred)
        
        logits_list = [global_logs]
        for l in li:
            r = torch.flatten(self.avg(l),1)
            c = self.classifier_patch(r)
            logits_list.append(c)
        combined_logits = 0
        for r in logits_list:
            combined_logits += r
        return logits_list,combined_logits



class exchange(nn.Module):
    def __init__(self,scale1,scale2,k_1,k_2):
        super().__init__()
        self.layers1 = [scale1,scale1+k_1+k_1,scale1+2*k_1+k_1,scale1+3*k_1+k_1,scale1+4*k_1+k_1]
        self.layers2 = [scale2,scale2+k_2+k_1,scale2+2*k_2+k_1,scale2+3*k_2+k_1,scale2+4*k_2+k_1]
        self.x1 = nn.Sequential(nn.Conv2d(scale1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y1 = nn.Sequential(nn.Conv2d(scale2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x1c = nn.Sequential(nn.Conv2d(scale1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y1t = nn.Sequential(nn.ConvTranspose2d(scale2,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        
        self.x2_input = nn.Sequential(nn.Conv2d(self.layers1[1],k_1,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        self.y2_input = nn.Sequential(nn.Conv2d(self.layers2[1],k_2,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        
        
        self.x2 = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y2 = nn.Sequential(nn.Conv2d(k_2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x2c = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y2t = nn.Sequential(nn.ConvTranspose2d(k_2,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        
        self.x3_input = nn.Sequential(nn.Conv2d(self.layers1[2],k_1,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        self.y3_input = nn.Sequential(nn.Conv2d(self.layers2[2],k_2,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        
        
        self.x3 = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y3 = nn.Sequential(nn.Conv2d(k_2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x3c = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y3t = nn.Sequential(nn.ConvTranspose2d(k_2,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        
        self.x4_input = nn.Sequential(nn.Conv2d(self.layers1[3],k_1,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        self.y4_input = nn.Sequential(nn.Conv2d(self.layers2[3],k_2,kernel_size=1,stride=1,padding=0,bias=False),nn.ReLU(inplace=True))
        
        self.x4 = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y4 = nn.Sequential(nn.Conv2d(k_2,k_2,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.x4c = nn.Sequential(nn.Conv2d(k_1,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        self.y4t = nn.Sequential(nn.ConvTranspose2d(k_2,k_1,kernel_size=3,stride=1,padding=1,bias=False),nn.ReLU(inplace=True))
        
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
        return x

class dfrag(nn.Module):
    def __init__(self,inplace,num_classes):
        super().__init__()
        self.net = VGGnet(inplace)
        self.avg = nn.AdaptiveAvgPool2d(1)
        
        self.conv1_stream1 = self._conv(1,64)
        self.conv1_stream2 = self._conv(1,64)
        self.conv1_stream3 = self._conv(1,64)
        self.conv1_stream4 = self._conv(1,64)
        self.conv1_stream5 = self._conv(1,64)
        self.maxp1 = nn.MaxPool2d(2,stride=(2,2))
        
        self.conv2_stream1 = self._conv(64,128)
        self.conv2_stream2 = self._conv(64,128)
        self.conv2_stream3 = self._conv(64,128)
        self.conv2_stream4 = self._conv(64,128)
        self.conv2_stream5 = self._conv(64,128)
        
        self.conv3_stream1 = self._conv(128,256)
        self.conv3_stream2 = self._conv(128,256)
        self.conv3_stream3 = self._conv(128,256)
        self.conv3_stream4 = self._conv(128,256)
        self.conv3_stream5 = self._conv(128,256)
        self.conv4_stream1 = self._conv(256,256)
        self.conv4_stream2 = self._conv(256,256)
        self.conv4_stream3 = self._conv(256,256)
        self.conv4_stream4 = self._conv(256,256)
        self.conv4_stream5 = self._conv(256,256)
        self.maxp4 = nn.MaxPool2d(2,stride=2) 
        self.maxp2 = nn.MaxPool2d(2,stride=2)
        self.maxp3 = nn.MaxPool2d(2,stride=2)
        
        self.classifier = nn.Linear(512,num_classes)
        self.exchange12_one = exchange(64,64,16,16)
        self.exchange23_one = exchange(64,64,16,16)
        self.exchange34_one = exchange(64,64,16,16)
        self.exchange45_one = exchange(64,64,16,16)
        
        self.exchange12_2 = exchange(128,128,32,32)
        self.exchange23_2 = exchange(128,128,32,32)
        self.exchange34_2 = exchange(128,128,32,32)
        self.exchange45_2 = exchange(128,128,32,32)
        
        self.exchange12_3 = exchange(256,256,32,32)
        self.exchange23_3 = exchange(256,256,32,32)
        self.exchange34_3 = exchange(256,256,32,32)
        self.exchange45_3 = exchange(256,256,32,32)
        
        
        self.exchange12_4 = exchange(256,256,16,16)
        self.exchange23_4 = exchange(256,256,16,16)
        self.exchange34_4 = exchange(256,256,16,16)
        self.exchange45_4 = exchange(256,256,16,16)
        
        
        
        
        self.classifier_global = nn.Linear(512,num_classes)
        self.classifier_patch = nn.Linear(256,num_classes) 
    
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
    
    def forward(self,x):
        x_global = self.net(x)
        step = 16
        
        #print(xlist[0].shape)
        xpatch = []
        # input image
        reslist = []
        for n in range(0,65,step):
            #print(x[:,:,:,n:n+64].shape)
            xpatch.append(x[:,:,:,n:n+64])
            
        x1 = self.conv1_stream1(xpatch[0])
        x1 = self.maxp1(x1)
        x2 = self.conv1_stream2(xpatch[1])
        x2 = self.maxp1(x2)
        x3 = self.conv1_stream3(xpatch[2])
        x3= self.maxp1(x3)
        x4 = self.conv1_stream4(xpatch[3])
        x4 = self.maxp1(x4)
        x5 = self.conv1_stream5(xpatch[4])
        x5 = self.maxp1(x5)
        
        x1,x2 = self.exchange12_one((x1,x2))
        x2,x3 = self.exchange23_one((x2,x3))
        x3,x4 = self.exchange34_one((x3,x4))
        x4,x5 = self.exchange45_one((x4,x5))
        
        x1 = self.conv2_stream1(x1)
        x1 = self.maxp2(x1)
        x2 = self.conv2_stream2(x2)
        x2 = self.maxp2(x2)
        x3 = self.conv2_stream3(x3)
        x3= self.maxp2(x3)
        x4 = self.conv2_stream4(x4)
        x4 = self.maxp2(x4)
        x5 = self.conv2_stream5(x5)
        x5 = self.maxp2(x5)
        
        x1,x2 = self.exchange12_2((x1,x2))
        x2,x3 = self.exchange23_2((x2,x3))
        x3,x4 = self.exchange34_2((x3,x4))
        x4,x5 = self.exchange45_2((x4,x5))
        
        
        x1 = self.conv3_stream1(x1)
        x1 = self.maxp3(x1)
        x2 = self.conv3_stream2(x2)
        x2 = self.maxp3(x2)
        x3 = self.conv3_stream3(x3)
        x3= self.maxp3(x3)
        x4 = self.conv3_stream4(x4)
        x4 = self.maxp3(x4)
        x5 = self.conv3_stream5(x5)
        x5 = self.maxp3(x5)
        
        x1,x2 = self.exchange12_3((x1,x2))
        x2,x3 = self.exchange23_3((x2,x3))
        x3,x4 = self.exchange34_3((x3,x4))
        x4,x5 = self.exchange45_3((x4,x5))
        
        x1 = self.conv4_stream1(x1)
        x1 = self.maxp4(x1)
        x2 = self.conv4_stream2(x2)
        x2 = self.maxp4(x2)
        x3 = self.conv4_stream3(x3)
        x3= self.maxp4(x3)
        x4 = self.conv4_stream4(x4)
        x4 = self.maxp4(x4)
        x5 = self.conv4_stream5(x5)
        x5 = self.maxp4(x5)
        
        x1,x2 = self.exchange12_4((x1,x2))
        x2,x3 = self.exchange23_4((x2,x3))
        x3,x4 = self.exchange34_4((x3,x4))
        x4,x5 = self.exchange45_4((x4,x5))
        
        li = [x1,x2,x3,x4,x5]
        global_pred = torch.flatten(self.avg(x_global),1)
        global_logs = self.classifier_global(global_pred)
        
        logits_list = [global_logs]
        for l in li:
            r = torch.flatten(self.avg(l),1)
            c = self.classifier_patch(r)
            logits_list.append(c)
        combined_logits =0
        for r in logits_list:
            combined_logits += r
        return logits_list,combined_logits


