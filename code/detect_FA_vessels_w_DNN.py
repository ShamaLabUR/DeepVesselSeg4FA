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


import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import functools
print = functools.partial(print,flush=True)
import numpy as np
import cv2 

from torchvision import transforms
from datasets.transforms_samples import ToTensorSample
from datasets import PatchData,RECOVERY
from models import create_model
import utils


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--data_dir', type=str, required=True)
parser.add_argument('-p','--pretrained_model_dir', type=str, required=True)
parser.add_argument('-i','--pretrained_model_id', type=int, required=True)
parser.add_argument('-s','--save_dir', type=str, required=True)
parser.add_argument('-b','--batch_size', type=int, default=16)
opt = parser.parse_args()

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transforms_ =  ToTensorSample()

test_data = RECOVERY(opt.data_dir,include_label=False,transforms=transforms_)

# model
print('creating model')
model = create_model('UNetVgg').to(device)
model = nn.DataParallel(model)
pretrained_model_name = "model_"+str(opt.pretrained_model_id)+".pth"
pretrained_model = os.path.join(opt.pretrained_model_dir,pretrained_model_name)
utils.load_checkpoint(pretrained_model,model)

image2patch = utils.Image2Patch(patch_size=(512,512),stride_size=(128,128))

model.eval()
with torch.no_grad():
    image = test_data.get_image_from_cross_val_id(opt.pretrained_model_id)["i"].unsqueeze(0)

    all_patches = image2patch.decompose(image)
    patch_loader = DataLoader(PatchData(all_patches),batch_size=opt.batch_size,shuffle=False)
    output = []
    for ind_patch,patches in enumerate(patch_loader):
        patches = patches.to(device)
        output.append( torch.sigmoid(model(patches)) )
     
    output = torch.cat(output)
    output = image2patch.compose(output.cpu()) # <1x1xHxW> 
    output = output.cpu().detach().numpy()

    save_name = test_data.get_filename_from_cross_val_id(opt.pretrained_model_id)[0]
    _,save_name,_ = utils.fileparts(save_name)

    # save image
    save_name_img = os.path.join(opt.save_dir,"predicted_"+save_name+".png")
    image_np = 65535*output.squeeze(0).squeeze(0)
    cv2.imwrite(save_name_img,image_np.astype(np.uint16))
