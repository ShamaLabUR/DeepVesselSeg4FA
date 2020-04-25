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
import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import disk,binary_dilation,skeletonize
from skimage import measure

def fileparts(full_name):
    path, filename = os.path.split(full_name)
    filename,ext = os.path.splitext(filename)
    return path,filename,ext

def load_checkpoint(checkpoint_path, model, optimizer=None):
    state = torch.load(checkpoint_path)
    #print(state['state_dict'])
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

class Image2Patch(object):
    def __init__(self,patch_size,stride_size):
        self.patch_size = patch_size
        self.stride_size = stride_size

    def _paint_border_overlap(self,image):
        patch_h,patch_w = self.patch_size 
        stride_h,stride_w = self.stride_size 
        img_h, img_w = image.shape[2],image.shape[3]
        
        left_h = (img_h - patch_h) % stride_h
        left_w = (img_w - patch_w) % stride_w

        pad_h = stride_h - left_h if left_h > 0 else 0
        pad_w = stride_w - left_w if left_w > 0 else 0

        padding = nn.ZeroPad2d((0,pad_w,0,pad_h))
        image = padding(image)
        self.new_size = (image.shape[2],image.shape[3])
        return image

    def _extract_order_overlap(self,image):
        patch_h,patch_w = self.patch_size 
        stride_h,stride_w = self.stride_size 
        N_img, ch, img_h, img_w = image.shape
    

        assert((img_h-patch_h)%stride_h == 0 and (img_w-patch_w)%stride_w == 0)
        N_patch_imgs = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)
        N_patch_total = N_patch_imgs * N_img
        

        patches = torch.zeros((N_patch_total,ch,patch_h,patch_w),dtype=torch.float32)
        count = 0
        for i in range(N_img):
            for h in range((img_h-patch_h)//stride_h+1):
                for w in range((img_w-patch_w)//stride_w+1):
                    patch = image[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                    patches[count] = patch
                    count += 1
        assert(count == N_patch_total)
        return patches

    def decompose(self,image):
        """
        Args:
            image: <B_i x C x H_i x W_i>
        Returns:
            patches: <B_p x C x H_p x W_p>
        """
        assert(len(image.shape)==4)
        self.image_size = (image.shape[2],image.shape[3])
        image = self._paint_border_overlap(image)     
        patches = self._extract_order_overlap(image)

        self.decomposed = True
        return patches

    def compose(self,patches):
        assert(self.decomposed==True)
        ch,patch_h,patch_w = patches.shape[1],patches.shape[2],patches.shape[3]
        img_h,img_w = self.new_size
        stride_h,stride_w = self.stride_size 

        N_patches_h = (img_h-patch_h)//stride_h+1
        N_patches_w = (img_w-patch_w)//stride_w+1
        N_patches_img = N_patches_h * N_patches_w

        assert(patches.shape[0]%N_patches_img==0)
        N_img = patches.shape[0]//N_patches_img

        full_prob = torch.zeros((N_img,ch,img_h,img_w))        
        full_sum = torch.zeros((N_img,ch,img_h,img_w))        

        count = 0
        for i in range(N_img):
            for h in range((img_h-patch_h)//stride_h+1):
                for w in range((img_w-patch_w)//stride_w+1):  
                    full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=patches[count]
                    full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
                    count += 1
 
        assert(count==patches.shape[0])
        image = full_prob / full_sum
        image = image[:,:,:self.image_size[0],:self.image_size[1]]
        self.decomposed = False
        return image


def dc_from_prrc(pr,rc):
    """
    Compute dice coefficient from precision and recall
    pr: precision
    rc: recall
    """
    dc = 2 * pr * rc / (pr + rc)
    return dc

def quant_16(x):
    x = (65535*x).astype(np.uint16)
    x = (x/65535.).astype(np.float32)
    return x


def CAL(pred,gt,alpha=2,beta=2):
    """
    Implementation of CAL metrics
    Reference:
    Gegúndez-Arias, Manuel Emilio, et al. 
    "A function for quality evaluation of retinal vessel segmentations." 
    IEEE transactions on medical imaging 31.2 (2011): 231-239.
    
    Args:
        im: predicted vessel map, binary, np.bool
        gt: ground truth vessel map, binary, np.bool
        alpha and beta: radii for structuring elements. See above reference for details
    """
    assert(pred.dtype==np.dtype(np.bool)), 'Input (prediction) must be binary array (np.bool)'
    assert(gt.dtype==np.dtype(np.bool)), 'Input (ground truth) must be binary array (np.bool)'
    
    # Connectivity (C)
    n_cc_pred = measure.label(pred).max()
    n_cc_gt = measure.label(gt).max()
    C = 1 - min(1, abs(n_cc_pred - n_cc_gt)/gt.sum() );
    
    # Area (A)
    se_a = disk(alpha)

    pred_dilated = binary_dilation(pred,se_a)
    gt_dilated = binary_dilation(gt,se_a)
    
    intersection = (pred_dilated&gt) | (pred&gt_dilated)
    union = pred | gt
    A = intersection.sum() / union.sum()
    
    # Length of Skeleton (L)
    se_b = disk(beta)
    
    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)
    
    pred_dilated = binary_dilation(pred,se_b)
    gt_dilated = binary_dilation(gt,se_b)
    
    intersection = (pred_skel&gt_dilated) | (pred_dilated&gt_skel)
    union = pred_skel | gt_skel
    L = intersection.sum() / union.sum()

    f = C*A*L
    return f
