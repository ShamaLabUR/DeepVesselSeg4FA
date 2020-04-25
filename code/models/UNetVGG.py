"""
The code is copyrighted by the authors. Permission to copy and use
this software for noncommercial use is hereby granted provided: (a)
this notice is retained in all copies, (2) the publication describing
the method (indicated below) is clearly cited, and (3) the
distribution from which the code was obtained is clearly cited. For
all other uses, please contact the authors.
 
The software code is provided "as is" with ABSOLUTELY NO WARRANTY
expressed or implied. Use at your own risk.

The code and the pre-trained deep neural network model provided with
this repository allow one to perform vessel detection in fluorescein
angiography images and to compute various evaluation metrics for the
detected vessel maps by comparing these against provided ground
truth. The related methodology and metrics are described in the paper:

L. Ding, M. H. Bawany, A. E. Kuriyan, R. S. Ramchandran, C. C. Wykoff, 
and G. Sharma, ``A novel deep learning pipeline for retinal vessel 
detection in fluorescein angiography,'' IEEE Trans. Image Proc., 
vol. 29, no. 1, 2020, accepted for publication, to appear.

Contacts: 
Li Ding: l.ding@rochester.edu
Gaurav Sharma: gaurav.sharma@rochester.edu
"""

from .model_utils import * 
from torchvision import models

class UNetVgg(nn.Module):
    def __init__(self,bilinear=True,pretrained=True):
        super(UNetVgg,self).__init__()
        self.encoder = models.vgg13(pretrained=pretrained).features
        self.inc = self.encoder[0:4] # 3 -- 64
        self.down1 = self.encoder[4:9] # 64 -- 128
        self.down2 = self.encoder[9:14] # 128 -- 256
        self.down3 = self.encoder[14:19] # 256 -- 512
        self.down4 = self.encoder[19:24] # 512 -- 512
        self.up1 = up(1024, 256,bilinear)
        self.up2 = up(512, 128,bilinear)
        self.up3 = up(256, 64,bilinear)
        self.up4 = up(128, 64,bilinear)
        self.outc = outconv(64,1)

    def forward(self, x):
        x1 = self.inc(x)    # 64
        x2 = self.down1(x1) # 128
        x3 = self.down2(x2) # 256
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 512
        x = self.up1(x5, x4) # 1024 -- 256 
        x = self.up2(x, x3) # 512 -- 128
        x = self.up3(x, x2) # 256 -- 64
        x = self.up4(x, x1) # 128 -- 64
        x = self.outc(x)    # 64 -- 1
        return x
