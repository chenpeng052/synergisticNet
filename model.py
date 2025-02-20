import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
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

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
          
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))
        self.skipcat = nn.Conv2d(5, 5, [1, 5], 1, 0)

    def forward(self, x, mask=None):
        outputs = []
        for n1,(attention, mlp) in enumerate(self.layers):
            x1 = attention(x, mask=mask)  # go to attention
            x1 = mlp(x1)  # go to MLP_Block
            outputs.append(x1)

        # Concatenate the outputs along the channel dimension and apply the convolution
        x1 = torch.cat((x.unsqueeze(3), outputs[0].unsqueeze(3),outputs[1].unsqueeze(3), outputs[2].unsqueeze(3),outputs[3].unsqueeze(3)), dim=3)
        x1 = self.skipcat(x1)
        x1 = x1.squeeze(3)

        attention, mlp = self.layers[1]
        x1 = attention(x1, mask=mask)
        x1 = mlp(x1)

        return x1

class TBFE(nn.Module):
    def __init__(self,input_channels ,reduction_N = 32):
        super(TBFE, self).__init__()
        self.point_wise = nn.Conv2d(input_channels,reduction_N,kernel_size=1,padding=0,bias=False)    
        self.depth_wise = nn.Sequential(nn.Conv2d(reduction_N, reduction_N, kernel_size=(3, 3),padding=1),nn.BatchNorm2d(reduction_N),nn.ReLU(),)

        self.conv3D = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1,1,3),padding=(0,0,1),stride=(1,1,1),bias=False)
        self.bn = nn.BatchNorm2d(reduction_N)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x_1 = self.point_wise(x)  
        x_2 = self.depth_wise(x_1)       
        x_2=x_1+x_2
        
        #DSC
        x_3 = x_1.unsqueeze(1)
        x_3 = self.conv3D(x_3)
        x_3 = x_3.squeeze(1)
        x = torch.cat((x_2,x_3),dim=1)
        
        return x

class HPA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(HPA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.map = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  #Y avg
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  #X avg
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))  #Y avg
        self.max_w = nn.AdaptiveMaxPool2d((1, None))  #X avg

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w  --->2048,2,11,11
        x_h = self.pool_h(group_x) #2048,2,11,1
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) #2048,2,1,11--->2048,2,11,1
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) #2048,2,22,1
        x_h, x_w = torch.split(hw, [h, w], dim=2) #2048,2,11,1
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) #2048,2,11,11
        x2 = self.conv3x3(group_x)  #2048,2,11,11

        y_h = self.max_h(group_x) #2048,2,11,1
        y_w = self.max_w(group_x).permute(0, 1, 3, 2)
        yhw = self.conv1x1(torch.cat([y_h, y_w], dim=2)) #2048,2,22,1
        y_h, y_w = torch.split(yhw, [h, w], dim=2) #2048,2,11,1
        y1 = self.gn(group_x * y_h.sigmoid() * y_w.permute(0, 1, 3, 2).sigmoid()) #2048,2,11,11
        y11 = y1.reshape(b * self.groups, c // self.groups, -1) # b*g, c//g, hw 2048,2,121
        y12 = self.softmax(self.map(y1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) #2048,1,2

        x11 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw 2048,2,121
        x12 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) #2048,2,1,1-->2048,2,1--->2048,1,2
        x21 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw  #2048,2,121
        x22 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) #2048,2,1,1-->2048,2,1--->2048,1,2
        weights = (torch.matmul(x12, y11) + torch.matmul(y12, x11)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)        
    
NUM_CLASS = 20

class synergisticNet(nn.Module):
    def __init__(self, input_channels=30, n_band_1=1584,  n_band_2=64 ,n_band_3=64,num_classes=NUM_CLASS, 
                 num_tokens=4, dim=64, depth=4, heads=16, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(synergisticNet, self).__init__()
        self.L = num_tokens
        self.cT = dim

        #TBFE_1
        self.lwm = TBFE(input_channels,reduction_N = 64)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        #TBFE_2
        self.lwm2 = TBFE(128)
        self.bn2 = nn.BatchNorm2d(64)

        self.hpa = HPA(channels=64)

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)


        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x):
        
        # TBFE_1
        x1 = self.lwm(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        # TBFE_2
        x2 = self.lwm2(x1)
        x2 = self.bn2(x2)
        x= self.relu(x2)
        
        # HPA
        x = self.conv2d_features(x)
        x = self.hpa(x)  

        x = rearrange(x,'b c h w -> b (h w) c')
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        #x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)  # main game
        
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x


if __name__ == '__main__':
    model = model()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 13, 13)
    y = model(input)
    print(y.size())

