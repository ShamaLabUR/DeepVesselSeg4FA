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
from torch import nn

class netD(nn.Module):
    def __init__(self,input_shape,n_feat=32):
        # input_shape: <CXHXW>
        super(netD,self).__init__()
        in_ch = input_shape[0]
        self.layer1 = double_conv(in_ch,n_feat)
        self.layer2 = down(n_feat,n_feat*2)
        self.layer3 = down(n_feat*2,n_feat*4)
        self.layer4 = down(n_feat*4,n_feat*8)
        self.layer5 = down(n_feat*8,n_feat*16)
        self.activation = nn.ReLU(inplace=True)
        n_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(n_size,64)
        self.fc2 = nn.Linear(64,1)
        
    # generate input sample and forward to get shape
    def _get_conv_output(self,shape):
        inputs = torch.randn(1,*shape)
        output_feat = self._forward_features(inputs)
        n_size = output_feat.view(1,-1).size(1)
        return n_size

    def _forward_features(self,x):
        assert(len(x.shape)==4),'input to MapFeat must be a batch (normally batch_size = 1)'
        x = self.layer1(x)        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def forward(self,x):
        x = self._forward_features(x)
        x = x.view(x.size(0),-1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
