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

from .UNetVGG import UNetVgg
from .Discriminator import netD

def create_model(arch):
    print('creating model: '+str(arch))
    if arch == 'UNetVgg':
        model = UNetVgg(pretrained=True)
    elif arch == 'netD':
        model = netD((4,256,256))
    else:
        raise ValueError('invalid model: ' + arch)
    return model

def create_GAN(arch_G,arch_D):
    print('creating GAN, Generator: ' + arch_G + ' Discriminator: ' + arch_D)
    net_G = create_model(arch_G)
    net_D = create_model(arch_D)
    return net_G,net_D
