import torch
from torch import nn
from torch.nn import functional as F
import copy
from einops import rearrange , repeat
import math
from timm.models.layers import DropPath , to_2tuple , trunc_normal_
import torch.utils.checkpoint as checkpoint


class Spatial_Enhance_Module(nn.Module):
    def __init__(self , in_channels , inter_channels=None , size=None):
        """Implementation of SAEM: Spatial Enhancement Module
        args:
            in_channels: original channel size
            inter_channels: channel size inside the block if not specifed reduced to half
        """
        super(Spatial_Enhance_Module , self).__init__()

        self.in_channels=in_channels
        self.inter_channels=inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels=in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels=1

        # dimension == 2
        conv_nd=nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn=nn.BatchNorm2d

        self.g=conv_nd(in_channels=self.in_channels , out_channels=self.inter_channels , kernel_size=1)
        self.W_z=nn.Sequential(
            conv_nd(in_channels=self.inter_channels , out_channels=self.in_channels , kernel_size=1) ,
            bn(self.in_channels)
        )

        # define Transformation 1 and 2
        self.T1=nn.Sequential(
            conv_nd(in_channels=self.in_channels , out_channels=self.inter_channels , kernel_size=1) ,
            bn(self.inter_channels) ,
            nn.Sigmoid()
        )
        self.T2=nn.Sequential(
            conv_nd(in_channels=self.in_channels , out_channels=self.inter_channels , kernel_size=1) ,
            bn(self.inter_channels) ,
            nn.Sigmoid()
        )

        self.dim_reduce=nn.Sequential(
            nn.Conv1d(
                in_channels=size * size ,
                out_channels=1 ,
                kernel_size=1 ,
                bias=False ,
            ) ,
        )

    def forward(self , x1 , x2):
        """
        args
            x: (N, C, H, W)
        """

        batch_size=x1.size(0)

        t1=self.T1(x1).view(batch_size , self.inter_channels , -1)
        t2=self.T2(x2).view(batch_size , self.inter_channels , -1)
        t1=t1.permute(0 , 2 , 1)
        Affinity_M=torch.matmul(t1 , t2)

        Affinity_M=Affinity_M.permute(0 , 2 , 1)  # B*HW*TF --> B*TF*HW
        Affinity_M=self.dim_reduce(Affinity_M)  # B*1*HW
        Affinity_M=Affinity_M.view(batch_size , 1 , x1.size(2) , x1.size(3))  # B*1*H*W

        x1=x1 * Affinity_M.expand_as(x1)

        return x1


class Spectral_Enhance_Module(nn.Module):
    def __init__(self , in_channels , in_channels2 , inter_channels=None , inter_channels2=None):
        """Implementation of SEEM: Spectral Enhancement Module
        args:
            in_channels: original channel size
            inter_channels: channel size inside the block
        """
        super(Spectral_Enhance_Module , self).__init__()

        self.in_channels=in_channels
        self.inter_channels=inter_channels
        self.in_channels2=in_channels2
        self.inter_channels2=inter_channels2

        if self.inter_channels is None:
            self.inter_channels=in_channels
            if self.inter_channels == 0:
                self.inter_channels=1
        if self.inter_channels2 is None:
            self.inter_channels2=in_channels2
            if self.inter_channels2 == 0:
                self.inter_channels2=1

        # dimension == 2
        conv_nd=nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn=nn.BatchNorm2d

        self.g=conv_nd(in_channels=self.in_channels , out_channels=self.inter_channels , kernel_size=1)
        self.W_z=nn.Sequential(
            conv_nd(in_channels=self.inter_channels , out_channels=self.in_channels , kernel_size=1) ,
            bn(self.in_channels)
        )

        # define Transformation 1 and 2
        self.T1=nn.Sequential(
            conv_nd(in_channels=self.in_channels , out_channels=self.inter_channels , kernel_size=1) ,
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            bn(self.inter_channels) ,
            nn.Sigmoid()
        )
        self.T2=nn.Sequential(
            conv_nd(in_channels=self.in_channels2 , out_channels=self.inter_channels2 , kernel_size=1) ,
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            bn(self.inter_channels2) ,
            nn.Sigmoid()
        )

        self.dim_reduce=nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels2 ,
                out_channels=1 ,
                kernel_size=1 ,
                bias=False ,
            )
        )

    def forward(self , x1 , x2):
        """
        args
            x: (N, C, H, W)
        """

        batch_size=x1.size(0)

        t1=self.T1(x1).view(batch_size , self.inter_channels , -1)
        t2=self.T2(x2).view(batch_size , self.inter_channels2 , -1)
        t2=t2.permute(0 , 2 , 1)
        Affinity_M=torch.matmul(t1 , t2)

        Affinity_M=Affinity_M.permute(0 , 2 , 1)  # B*C1*C2 --> B*C2*C1
        Affinity_M=self.dim_reduce(Affinity_M)  # B*1*C1
        Affinity_M=Affinity_M.view(batch_size , x1.size(1) , 1 , 1)  # B*C1*1*1

        x1=x1 * Affinity_M.expand_as(x1)

        return x1


class conv_bn_relu(nn.Module):
    def __init__(self , in_channels , out_channels , kernel_size , stride=1 ,
                 padding=0 , dilation=1 , groups=1 , bias=True):
        super(conv_bn_relu , self).__init__()
        self.conv=nn.Conv2d(in_channels , out_channels , kernel_size , stride ,
                            padding , dilation , groups , bias)
        self.bn=nn.BatchNorm2d(out_channels)
        self.activation=nn.ReLU()

    def forward(self , x):
        out=self.conv(x)
        out=self.bn(out)
        out=self.activation(out)

        return out



class conv_block(nn.Module):
    def __init__(self , in_channels , out_channels , kernel_size , stride=1 ,
                 padding=0 , dilation=1 , groups=1 , bias=True):
        super(conv_block , self).__init__()
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels , out_channels , kernel_size=1 , stride=stride , bias=bias) ,
            nn.BatchNorm2d(out_channels))
        self.conv1=nn.Conv2d(in_channels , out_channels , kernel_size , stride ,
                             padding , dilation , groups , bias)
        self.bn=nn.BatchNorm2d(out_channels)

        self.conv2=nn.Conv2d(out_channels , out_channels , kernel_size , stride ,
                             padding , dilation , groups , bias)
        self.bn2=nn.BatchNorm2d(out_channels)

        self.conv3=nn.Conv2d(out_channels , out_channels , kernel_size , stride ,
                             padding , dilation , groups , bias)
        self.bn3=nn.BatchNorm2d(out_channels)
        self.activation=nn.ReLU()

    def forward(self , x):
        identity=self.downsample(x)

        out=self.conv1(x)
        out=self.bn(out)
        out=self.activation(out)

        out=self.conv3(out)
        out=self.bn3(out)

        out=out + identity
        out=self.activation(out)
        return out



class HetConv(nn.Module):
    def __init__(self , in_channels , out_channels , kernel_size=3 , stride=1 , padding=None , bias=None , p=64 , g=64):
        super(HetConv , self).__init__()
        # Groupwise Convolution
        self.gwc=nn.Conv2d(in_channels , out_channels , kernel_size=kernel_size , groups=g , padding=kernel_size // 3 ,
                           stride=stride)
        # Pointwise Convolution
        self.pwc=nn.Conv2d(in_channels , out_channels , kernel_size=1 , groups=p , stride=stride)

    def forward(self , x):
        return self.gwc(x) + self.pwc(x)


class Mlp(nn.Module):
    def __init__(self , in_features , hidden_features=None , out_features=None , act_layer=nn.GELU , drop=0.):
        super().__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.fc1=nn.Linear(in_features , hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features , out_features)
        self.drop=nn.Dropout(drop)

    def forward(self , x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x


class GlobalFilter(nn.Module):
    def __init__(self , dim , h=14 , w=8):
        super().__init__()
        self.complex_weight=nn.Parameter(torch.randn(h , w , dim , 2 , dtype=torch.float32) * 0.02)
        self.w=w
        self.h=h

    def forward(self , x , spatial_size=None):
        B , N , C=x.shape
        if spatial_size is None:
            a=b=int(math.sqrt(N))
        else:
            a , b=spatial_size

        x=x.view(B , a , b , C)

        x=x.to(torch.float32)

        x=torch.fft.rfft2(x , dim=(1 , 2) , norm='ortho')
        weight=torch.view_as_complex(self.complex_weight)
        x=x * weight
        x=torch.fft.irfft2(x , s=(a , b) , dim=(1 , 2) , norm='ortho')

        x=x.reshape(B , N , C)

        return x


class Bi_direct_adapter(nn.Module):
    def __init__(self , dim=32 , xavier_init=False):
        super().__init__()

        self.adapter_down=nn.Linear(64 , dim)
        self.adapter_up=nn.Linear(dim , 64)
        self.adapter_mid=nn.Linear(dim , dim)

        # nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        # self.act = QuickGELU()
        self.dropout=nn.Dropout(0.1)
        self.dim=dim

    def forward(self , x):
        B , N , C=x.shape
        x_down=self.adapter_down(x)
        x_down=self.adapter_mid(x_down)
        x_down=self.dropout(x_down)
        x_up=self.adapter_up(x_down)
        return x_up


class Block(nn.Module):

    def __init__(self , dim , mlp_ratio=4. , drop=0. , drop_path=0. , act_layer=nn.GELU , norm_layer=nn.LayerNorm ,
                 h=14 , w=8):
        super().__init__()
        self.norm1=norm_layer(dim)
        self.filter=GlobalFilter(dim , h=h , w=w)
        self.drop_path=DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2=norm_layer(dim)
        mlp_hidden_dim=int(dim * mlp_ratio)
        self.mlp=Mlp(in_features=dim , hidden_features=mlp_hidden_dim , act_layer=act_layer , drop=drop)

        self.adap_t=Bi_direct_adapter()
        self.adap2_t=Bi_direct_adapter()

    def forward(self , x , xi):
        xoi=x
        x_attn=self.filter(self.norm1(x))
        x=x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))

        xi_attn=self.filter(self.norm1(xi))
        xi=xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xoi)))

        xori=x
        x=x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(xi)))
        xi=xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))

        return x , xi


class SqueezeAndExcitation(nn.Module):
    def __init__(self , channel ,
                 reduction=32 , activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation , self).__init__()
        self.fc=nn.Sequential(
            nn.Conv2d(channel , reduction , kernel_size=1) ,
            activation ,
            nn.Conv2d(reduction , channel , kernel_size=1) ,
            nn.Sigmoid()
        )

    def forward(self , x):
        weighting=F.adaptive_avg_pool2d(x , 1)
        weighting=self.fc(weighting)
        y=x * weighting
        return y


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self , channels_in , activation=nn.ReLU()):
        super(SqueezeAndExciteFusionAdd , self).__init__()

        self.se_rgb=SqueezeAndExcitation(channels_in ,
                                         activation=activation)
        self.se_depth=SqueezeAndExcitation(channels_in ,
                                           activation=activation)

    def forward(self , rgb , depth):
        rgb=self.se_rgb(rgb)
        depth=self.se_depth(depth)
        out=rgb + depth
        return out


class BPFTNet(nn.Module):

    def __init__(self , input_channels , input_channels2 , n_classes , patch_size , bn_threshold):
        super(BPFTNet , self).__init__()

        self.activation=nn.ReLU(inplace=True)
        self.avg_pool=nn.AdaptiveAvgPool2d((1 , 1))
        self.planes_a=[128 , 64 , 32]
        self.planes_b=[8 , 16 , 32]

        self.ratio=4
        self.conv5=nn.Sequential(
            nn.Conv3d(1 , 8 , (9 , 3 , 3) , padding=(0 , 1 , 1) , stride=1) ,
            nn.BatchNorm3d(8) ,
            nn.ReLU()
        )

        self.conv6=nn.Sequential(
            HetConv(8 * (input_channels - 8) , 16 * self.ratio ,
                    p=1 ,
                    g=(16 * self.ratio) // 4 if (8 * (input_channels - 8)) % 16 == 0 else (16 * self.ratio) // 8 ,
                    ) ,
            nn.BatchNorm2d(16 * self.ratio) ,
            nn.ReLU()
        )

        self.lidarConv=nn.Sequential(
            nn.Conv2d(input_channels2 , 16 * self.ratio , 3 , 1 , 1) ,
            nn.BatchNorm2d(16 * self.ratio) ,
            nn.ReLU()
        )
        self.SFF=SqueezeAndExciteFusionAdd(16 * self.ratio)

        self.conv1_a=conv_bn_relu(input_channels , self.planes_a[0] , kernel_size=3 , padding=1 , bias=True)
        self.conv1_b=conv_bn_relu(input_channels2 , self.planes_b[0] , kernel_size=3 , padding=1 , bias=True)

        self.conv2_a=conv_block(self.planes_a[0] , self.planes_a[1] , kernel_size=3 , padding=1 , bias=True)
        self.conv2_b=conv_block(self.planes_b[0] , self.planes_b[1] , kernel_size=3 , padding=1 , bias=True)

        self.conv3_a=conv_block(self.planes_a[1] , self.planes_a[2] , kernel_size=3 , padding=1 , bias=True)
        self.conv3_b=conv_block(self.planes_b[1] , self.planes_b[2] , kernel_size=3 , padding=1 , bias=True)

        self.SAEM=Spatial_Enhance_Module(in_channels=self.planes_a[2] , inter_channels=self.planes_a[2] // 2 ,
                                        size=patch_size)
        self.SEEM=Spectral_Enhance_Module(in_channels=self.planes_b[2] , in_channels2=self.planes_a[2])

        self.dim=64
        self.num_patches=patch_size * patch_size

        self.conv_block1=nn.Sequential(
            nn.Conv2d(input_channels2 + input_channels , 64 , kernel_size=3 , stride=1 , padding=1) ,
            nn.BatchNorm2d(64) ,
            nn.ReLU()
        )

        self.conv_block2=nn.Sequential(
            nn.Conv2d(self.planes_a[0] + self.planes_b[0] , 64 , kernel_size=3 , stride=1 , padding=1) ,
            nn.BatchNorm2d(64) ,
            nn.ReLU()
        )

        self.conv_block3=nn.Sequential(
            nn.Conv2d(self.planes_a[1] + self.planes_b[1] , 64 , kernel_size=3 , stride=1 , padding=1) ,
            nn.BatchNorm2d(64) ,
            nn.ReLU()
        )


        self.fc_embedding=nn.Linear(input_channels2 + input_channels , 64)
        self.position_embeddings=nn.Parameter(torch.randn(1 , self.num_patches , self.dim))

        self.position_embeddings2=nn.Parameter(torch.randn(1 , self.num_patches , self.dim))
        self.cls_token=nn.Parameter(torch.zeros(1 , 1 , self.dim))
        self.pos_drop=nn.Dropout(0.1)
        self.act=nn.ReLU()

        self.inner_dim=200

        drop_path_rate=0.1
        depth=6
        dpr=[drop_path_rate for _ in range(depth)]
        self.FourierTransformer=nn.ModuleList([
            Block(
                dim=self.dim , mlp_ratio=4 ,
                drop=0.1 , drop_path=dpr[i] , norm_layer=nn.LayerNorm , h=patch_size , w=patch_size // 2 + 1)
            for i in range(depth)])

        self.FusionLayer=nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim ,
                out_channels=32 ,
                kernel_size=1 ,
            ) ,
            nn.BatchNorm2d(32) ,
            nn.ReLU() ,
        )
        self.transfor=nn.Conv2d(patch_size * patch_size , self.dim , kernel_size=1)

        self.conv_block_MRFN=conv_block(self.dim , self.dim , kernel_size=3 , padding=1 , stride=1 , bias=True)

        self.cls_fc=nn.Linear(self.planes_a[2] * 2 , 64)
        self.fc=nn.Linear(self.dim , n_classes)
        self.fc_x=nn.Linear(self.dim , n_classes)
        self.fc_y=nn.Linear(self.dim , n_classes)

        for m in self.modules():
            if isinstance(m , nn.Conv2d):
                nn.init.kaiming_normal_(m.weight , mode='fan_out' , nonlinearity='relu')
            elif isinstance(m , (nn.BatchNorm2d , nn.GroupNorm)):
                nn.init.constant_(m.weight , 1)
                nn.init.constant_(m.bias , 0)

    def forward(self , x1 , x2):
        B , N , patchsize , _=x1.shape

        x=x1.reshape(x1.shape[0] , -1 , patchsize , patchsize)
        x=x.unsqueeze(1)
        x=self.conv5(x)
        x=x.reshape(x.shape[0] , -1 , patchsize , patchsize)

        x=self.conv6(x)

        l=self.lidarConv(x2)
        # x = x + l
        x=self.SFF(x , l)
        x_f=x
        

        x1=self.conv1_a(x1)
        x2=self.conv1_b(x2)

        x1=self.conv2_a(x1)
        x2=self.conv2_b(x2)

        x1=self.conv3_a(x1)
        x2=self.conv3_b(x2)

        ss_x1=self.SAEM(x1 , x2)
        ss_x2=self.SEEM(x2 , x1)
        y=torch.cat([ss_x1 , ss_x2] , 1)
        y_f=y
        y=y.flatten(2).transpose(-1 , -2)
        x=x.flatten(2).transpose(-1 , -2)
        x=self.pos_drop(x + self.position_embeddings2)
        y=self.pos_drop(y + self.position_embeddings)
        for blk in self.FourierTransformer:
            y , x=blk(y , x)
        y=x + y
        x=y.reshape(B , patchsize , patchsize , self.dim).permute(0 , 3 , 1 , 2)

        x=self.avg_pool(x)
        x=torch.flatten(x , 1)
        x=self.fc(x)

        x_f=self.avg_pool(x_f)
        x_f=torch.flatten(x_f , 1)
        x_f=self.fc_x(x_f)

        y_f=self.avg_pool(y_f)
        y_f=torch.flatten(y_f , 1)
        y_f=self.fc_y(y_f)
        return x_f , y_f , x