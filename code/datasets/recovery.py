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
import glob

import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class RECOVERY(Dataset):
    """
    RECOVERY-FA19 Dataset
    DOI: 10.21227/m9yw-xs04
    
    Args:
        root (string): Root directory of dataset where 
            ``Images_RECOVERY-FA19`` and  ``Labels_RECOVERY-FA19`` exist.
        transform (callable, optional): A function that takes in a dict object
            and returns a transformed version.
    """
    
    def __init__(self,root,include_label=True,transforms=None):
        self.root = os.path.expanduser(root)
        self.include_label = include_label
        self.transforms = transforms

        images_name = os.path.join(self.root,"Images_RECOVERY-FA19","*tif")
        self.image_filenames = sorted(glob.glob(images_name))
         
        if self.include_label:
            labels_name = os.path.join(self.root,"Labels_RECOVERY-FA19","*png")
            self.label_filenames = sorted(glob.glob(labels_name))
        
        self.n_img = len(self.image_filenames)

    def get_filename_from_cross_val_id(self,model_id):
        image_filename = [f for f in self.image_filenames if "Img0"+str(model_id) in f] 
        if self.include_label:
            label_filename = [f for f in self.label_filenames if "Label0"+str(model_id) in f] 
            return image_filename[0], label_filename[0]
        else:
            return image_filename[0],None

    def get_image_from_cross_val_id(self,model_id):
        image_filename = self.get_filename_from_cross_val_id(model_id)[0]
        image = Image.open(image_filename).convert('RGB')

        if self.include_label:
            label_filename = self._get_filename_for_cross_val[model_id][1]
            label = Image.open(label_filename).convert('L')
        else:
            label = Image.new('L',image.size)

        sample = {'i':image,'l':label}
        if self.transforms:
            sample = self.transforms(sample) 
        return sample        

    def __getitem__(self,index):
        image = Image.open(self.image_filenames[index]).convert('RGB')
        
        if self.include_label:
            label = Image.open(self.label_filenames[index]).convert('L')
        else:
            label = Image.new('L',image.size)
        
        sample = {'i':image,'l':label}
        if self.transforms:
            sample = self.transforms(sample) 
        return sample        
        
    def __len__(self):
        return self.n_img


#class RECOVERY_Results(Dataset):
#    def __init__(self,root):
#        self.root = root
#        results_name = os.path.join(self.root,"prediction_Img*.png")
#        self.result_filenames = sorted(glob.glob(results_name))
#
#        self.n_img = len(self.result_filenames)
#
#    def __getitem__(self,index):
#        
#
#    def __len__(self):
#        return self.n_img
