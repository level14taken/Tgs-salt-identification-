import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import io
import PIL
import matplotlib.pyplot as plt
#from albumentations import Compose,PadIfNeeded,RandomBrightness,Normalize

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x         
        
        
        
        
class ChannelAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
                 )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class SpatialAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttentionGate, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.cg= ChannelAttentionGate(out_channels)
        self.sg= SpatialAttentionGate(out_channels)
        
    def forward(self, x ):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        g1= self.sg(x)
        g2= self.cg(x)
        x= g1*x+g2*x
        return x
        
        
class UNetDPSV2(nn.Module):
   
    def __init__(self ):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=False)

        self.conv_one = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )# 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  #128
        self.encoder4 = self.resnet.layer3  #256
        self.encoder5 = self.resnet.layer4  #512

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(512+256, 512, 64)
        self.decoder4 = Decoder(64+256, 256, 64)
        self.decoder3 = Decoder(64+128, 128,  64)
        self.decoder2 = Decoder( 64+ 64, 64, 64)
       
        self.conv0= nn.Conv2d(256,1,1)
        self.pool= nn.AdaptiveAvgPool2d((1,1))
        self.dense=nn.Linear(512,1)


        self.logit    = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  1, kernel_size=1, padding=0),
        )


    def forward(self, x):
        #batch_size,C,H,W = x.shape

        x = self.conv_one(x)

        e2 = self.encoder2( x)  #; print('e2',e2.size())
        e3 = self.encoder3(e2)  #; print('e3',e3.size())
        e4 = self.encoder4(e3)  #; print('e4',e4.size())
        e5 = self.encoder5(e4)  #; print('e5',e5.size())
                            #; print('center',f.size())
        f = self.center(e5)
        # print(e5.shape,f.shape)
        
        d5 = self.decoder5(torch.cat([f, e5], 1))  #; print('d5',d5.size(),f.size())
        d4 = self.decoder4(torch.cat([d5, e4], 1))  #; print('d4',d4.size())
        d3= self.decoder3(torch.cat([d4, e3], 1))  #; print('d3',d3.size())
        d2 = self.decoder2(torch.cat([d3, e2], 1))  #; print('d2',d2.size())
        
        d2 = torch.cat((d2,
          F.upsample(d3,scale_factor=2,mode='bilinear',align_corners=False),
          F.upsample(d4,scale_factor=4,mode='bilinear',align_corners=False),            
          F.upsample(d5,scale_factor=8,mode='bilinear',align_corners=False),          
                      ),1)
        

        final = self.logit(d2)                     #; print('logit',logit.size())
        no_mask= self.dense(self.pool(e5).view(-1,512))

        return [no_mask,final]
       
def get_model():
    model = UNetDPSV2()
    model.load_state_dict(torch.load(f'./binary_0.pth',map_location= 'cpu'),strict=False)
    model.eval()
    return model

def remove_smasks(array_):
  """ PostProcessing Removes small masks of 10 pixels, as it  is most likely a result of overfitting."""
  array_= list(array_)
  for ind,curr in enumerate(array_):  
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(curr, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 10
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    array_[ind]= img2 
  return np.array(array_)
