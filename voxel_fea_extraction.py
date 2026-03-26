import numpy as np
# import spconv
import spconv.pytorch as spconv
import torch
from torch import nn
from .voxel_multi import Feat_Encoder_Decoder

def extract_nonzero_features(x):
    device = x.device 
    nonzero_index = torch.sum(torch.abs(x), dim=1).nonzero()    
    coords = nonzero_index.type(torch.int32).to(device)     
    channels = int(x.shape[1])      
    features = x.permute(0, 2, 3, 4, 1).reshape(-1, channels)       
    features = features[torch.sum(torch.abs(features), dim=1).nonzero(), :]     
    features = features.squeeze(1).to(device)           
    coords, _, _ = torch.unique(coords, return_inverse=True, return_counts=True, dim=0)     
    return coords, features


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation, bn=False):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv3d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        if bn: 
            self.bn = nn.BatchNorm3d(planes)
        else:
            self.bn = bn
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        if self.bn:
            x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Voxel_fea_extract(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Voxel_fea_extract, self).__init__()
        
        sparse_shape = np.array(output_shape)
        self.sparse_shape = sparse_shape        

        # Completion sub-network
        mybias = False      # default: False
        chs = [32, 32, 48, 64]    

        self.a_conv1 = nn.Sequential(nn.Conv3d(chs[0], chs[1], 3, 1, padding=1, bias=mybias), 
                                     nn.ReLU()
                                    )
        
        dil = 4

        self.a_conv2 = nn.Sequential(nn.Conv3d(chs[0], chs[1], 3, 1, padding=1, bias=mybias), 
                                     nn.ReLU()
                                    )
        self.a_conv2_aspp = _ASPPModule(chs[0], chs[1], kernel_size=3, padding=dil, dilation=dil)
        
        self.a_conv3 = nn.Sequential(nn.Conv3d(chs[0], chs[1], 5, 1, padding=2, bias=mybias), 
                                     nn.ReLU()
                                    )
        self.a_conv3_aspp = _ASPPModule(chs[0], chs[1], kernel_size=3, padding=dil, dilation=dil)
        
        self.a_conv4 = nn.Sequential(nn.Conv3d(chs[0], chs[1], 7, 1, padding=3, bias=mybias), 
                                     nn.ReLU()
                                    )
        self.a_conv4_aspp = _ASPPModule(chs[0], chs[1], kernel_size=3, padding=dil, dilation=dil)
        
        self.a_conv5 = nn.Sequential(nn.Conv3d(chs[0], chs[2], 3, 1, padding=1, bias=mybias), 
                                     nn.ReLU()
                                    )
        self.a_conv5_aspp = _ASPPModule(chs[2], chs[2], kernel_size=3, padding=dil, dilation=dil)
        
        self.a_conv6 = nn.Sequential(nn.Conv3d(chs[0], chs[2], 5, 1, padding=2, bias=mybias), 
                                     nn.ReLU()
                                    )
        self.a_conv6_aspp = _ASPPModule(chs[2], chs[2], kernel_size=3, padding=dil, dilation=dil)
        
        self.a_conv7 = nn.Sequential(nn.Conv3d(chs[0], chs[2], 7, 1, padding=3, bias=mybias), 
                                     nn.ReLU()
                                    )
        self.a_conv7_aspp = _ASPPModule(chs[2], chs[2], kernel_size=3, padding=dil, dilation=dil)
        
        self.conv12 = nn.Conv3d((chs[1]+chs[2])*3, chs[2], 3, 1, padding=1, bias=mybias)

        self.c_conv1 = nn.Sequential(nn.Conv3d(chs[0], chs[3], 3, 1, padding=1, bias=mybias), 
                                     nn.ReLU()
                                    )
        self.c_conv1_aspp = _ASPPModule(chs[3], chs[3], kernel_size=3, padding = dil, dilation=dil)
        
        self.c_conv2 = nn.Sequential(nn.Conv3d(chs[0], chs[3], 5, 1, padding=2, bias=mybias), 
                                     nn.ReLU()
                                    )
        self.c_conv2_aspp = _ASPPModule(chs[3], chs[3], kernel_size=3, padding = dil, dilation=dil)
        
        self.c_conv3 = nn.Sequential(nn.Conv3d(chs[0], chs[3], 7, 1, padding=3, bias=mybias), 
                                     nn.ReLU())
        self.c_conv3_aspp = _ASPPModule(chs[3], chs[3], kernel_size=3, padding = dil, dilation=dil)
        
        self.conv123 = nn.Conv3d(chs[1]+chs[2]+chs[3]*3, chs[3], 3, 1, padding=1, bias=mybias)

        self.relu = nn.ReLU(inplace=True)
        
        self.feat_process = Feat_Encoder_Decoder(class_num=nclasses, input_dimensions=chs[3]*self.sparse_shape[2], out_dim_2D =self.sparse_shape[2])
        
    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        
        x_sparse = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)   
        
        # Spase to dense
        x_dense = x_sparse.dense()  
        
        x1 = self.a_conv1(x_dense)          
        
        a1 = self.a_conv2(x1)       
        a1_aspp = self.a_conv2_aspp(a1)     
        a1 = a1_aspp + a1           
        
        a2 = self.a_conv3(x1)       
        a2_aspp = self.a_conv3_aspp(a2)     
        a2 = a2_aspp + a2           

        a3 = self.a_conv4(x1)       
        a3_aspp = self.a_conv4_aspp(a3)     
        a3 = a3_aspp + a3           
        t1 = torch.cat((a1, a2, a3), 1)     
        
        b1 = self.a_conv5(x1)       
        b1_aspp = self.a_conv5_aspp(b1)     
        b1 = b1_aspp + b1           
        
        b2 = self.a_conv6(x1)       
        b2_aspp = self.a_conv6_aspp(b2)     
        b2 = b2_aspp + b2           

        b3 = self.a_conv7(x1)      
        b3_aspp = self.a_conv7_aspp(b3)     
        b3 = b3_aspp + b3           
        t2 = torch.cat((b1, b2, b3), 1)     

        t12 = torch.cat((t1, t2), 1)    

        context12 = self.conv12(t12)        
        context12 = self.relu(context12)    
                
        c1 = self.c_conv1(x1)               
        c1_aspp = self.c_conv1_aspp(c1)     
        c1 = c1_aspp + c1                   

        c2 = self.c_conv2(x1)               
        c2_aspp = self.c_conv2_aspp(c2)     
        c2 = c2_aspp + c2                   

        c3 = self.c_conv3(x1)               
        c3_aspp = self.c_conv3_aspp(c3)     
        c3 = c3_aspp + c3                   
        
        context123 = torch.cat((x_dense, context12, c1, c2, c3), dim=1)     
        context123 = self.conv123(context123)   
        context123 = self.relu(context123)      
        
        feat = context123.permute(0,1,4,2,3)        
        feat = feat.reshape(feat.shape[0], feat.shape[1]*feat.shape[2], feat.shape[3], feat.shape[4])       
        voxel_feat = self.feat_process(feat)       
        
        # Dense to sparse,  coord: (524288, 4), features: (524288, 64)
        coord, features = extract_nonzero_features(voxel_feat)
        
        return coord, features     
        