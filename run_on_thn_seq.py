from datasets import DatasetThnSeq # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from deeplabv3 import DeepLabV3

from utils import label_img_to_color

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

batch_size = 8

network = DeepLabV3("eval2", project_dir="/staging/frexgus/multitask").cuda()
network.load_state_dict(torch.load("/staging/frexgus/multitask/training_logs/model_13_2_2_2/checkpoints/model_13_2_2_2_epoch_947.pth"))

val_dataset = DatasetThnSeq()

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=2)

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
unsorted_img_ids = []
for step, (imgs, img_ids) in enumerate(val_loader):
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        ########################################################################
        # save data for visualization:
        ########################################################################
        outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
        pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)

        for i in range(pred_label_imgs.shape[0]):
            pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
            img_id = img_ids[i]
            img = imgs[i] # (shape: (3, img_h, img_w))

            img = img.data.cpu().numpy()
            img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
            img = img*np.array([0.229, 0.224, 0.225])
            img = img + np.array([0.485, 0.456, 0.406])
            img = img*255.0
            img = img.astype(np.uint8)

            pred_label_img_color = label_img_to_color(pred_label_img)
            overlayed_img = 0.35*img + 0.65*pred_label_img_color
            overlayed_img = overlayed_img.astype(np.uint8)

            img_h = overlayed_img.shape[0]
            img_w = overlayed_img.shape[1]

            cv2.imwrite("/staging/frexgus/multitask/training_logs/model_eval2/" + img_id + ".png", img)
            cv2.imwrite("/staging/frexgus/multitask/training_logs/model_eval2/" + img_id + "_pred.png", pred_label_img_color)
            cv2.imwrite("/staging/frexgus/multitask/training_logs/model_eval2/" + img_id + "_overlayed.png", overlayed_img)

            unsorted_img_ids.append(img_id)

out = cv2.VideoWriter("/staging/frexgus/multitask/training_logs/model_eval2/thn_combined.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12, (2*img_w, 2*img_h))

sorted_img_ids = sorted(unsorted_img_ids)
for img_id in sorted_img_ids:
    img = cv2.imread("/staging/frexgus/multitask/training_logs/model_eval2/" + img_id + ".png", -1)
    pred_img = cv2.imread("/staging/frexgus/multitask/training_logs/model_eval2/" + img_id + "_pred.png", -1)
    overlayed_img = cv2.imread("/staging/frexgus/multitask/training_logs/model_eval2/" + img_id + "_overlayed.png", -1)

    combined_img = np.zeros((2*img_h, 2*img_w, 3), dtype=np.uint8)

    combined_img[0:img_h, 0:img_w] = img
    combined_img[0:img_h, img_w:(2*img_w)] = pred_img
    combined_img[img_h:(2*img_h), (int(img_w/2)):(img_w + int(img_w/2))] = overlayed_img

    out.write(combined_img)

out.release()
