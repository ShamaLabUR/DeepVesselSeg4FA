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
import argparse
import glob
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve,roc_curve,auc
from datasets import PatchData,RECOVERY
import utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--data_dir', type=str, required=True)
parser.add_argument('-s','--save_dir', type=str, required=True)
opt = parser.parse_args()

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

test_data = os.path.join(opt.save_dir,"predicted_Img*.png")
test_data = sorted(glob.glob(test_data))
n_img = len(test_data)

outputs = []
gts = []
cals = []
for index in range(n_img):
    # load prediction
    current_pred_name = test_data[index]
    p,f,_ = utils.fileparts(current_pred_name)
    current_pred = cv2.imread(current_pred_name,cv2.IMREAD_UNCHANGED)

    # load ground truth
    current_gt_name = "Label"+f[13:]+".png" #f: predicted_Img*
    current_gt_name = os.path.join(opt.data_dir,"Labels_RECOVERY-FA19",current_gt_name)
    current_gt = cv2.imread(current_gt_name,cv2.IMREAD_GRAYSCALE)


    current_gt = current_gt[:,:].astype(np.float32) / 255
    current_pred = current_pred[:,:].astype(np.float32) / 65535
    current_gt_binary = current_gt>0
    current_pred_binary = current_pred>0.5
    
    # eval on every image
    y_true = current_gt.reshape(-1)
    y_scores = current_pred.reshape(-1)
    
    outputs.append(y_scores)
    gts.append(y_true)
    
    f = utils.CAL(current_pred_binary,current_gt_binary)
    cals.append(f)
    
    #precision,recall,th_prrc = precision_recall_curve(y_true,y_scores,pos_label=1)
    #prrc_auc = auc(recall,precision,True)

    #dc = utils.dc_from_prrc(precision,recall)
    #dc_max = np.nanmax(dc)
    #print('{}\t PRRC AUC {:.4f}, max DC: {:.4f}'.format(index,prrc_auc,dc_max))


cals = np.hstack(cals).mean()

y_true = np.hstack(gts)
y_scores = np.hstack(outputs)


# roc curve
fpr,tpr,th_roc = roc_curve(y_true, y_scores)
roc_auc_2 = auc(fpr,tpr,True)
print('ROC AUC: {:.4f}'.format(roc_auc_2))

# precision recall
precision,recall,th_prrc = precision_recall_curve(y_true,y_scores,pos_label=1)
prrc_auc = auc(recall,precision,True)
print('PRRC AUC: {:.4f}'.format(prrc_auc))

# dice coefficient
dc = utils.dc_from_prrc(precision,recall)
dc_max = np.nanmax(dc)
print('Max Dice coefficient: {:.4f}'.format(dc_max))

# CAL
print('CAL metric: {:.4f}'.format( cals))
