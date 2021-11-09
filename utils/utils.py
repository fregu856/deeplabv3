# camera-ready

import torch
import torch.nn as nn

import numpy as np

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]

# function for colorizing a label image:
def label_img_to_color(img):
    img=img.astype(np.uint8)
    lut=np.ones((256,1,3),dtype=np.uint8)
    
    ine = np.array([[
         [128, 64,128],#road 0 
         [244, 35,232],#sidewalk 1
         [ 70, 70, 70],#building 2
         [190,153,153],#wall  3
         [153,153,153],#fence 4
         [250,170, 30],#pole  5
         [220,220,  0],#traffic light 6
         [0,0, 255],#traffic sign 7
         [152,251,152],#vegetation 8
         [ 70,130,180],#terrain 9
         [220, 20, 60],#sky 10 
         [255,  0,  0],#person 11
         [  0,  0,142],#rider 12
         [  0,  0, 70],#car 13
         [  0, 60,100],#truck 14
         [  0, 80,100],#bus 15
         [  0,  0,230],#train 16
         [119, 11, 32],#motorcycle 17
         [81,  0, 81],#bicycle 18
         [0,0,0]#background 19
    ]])

    
    lut[0:20,:,:]=ine.reshape(20,1,3)
    img_color=cv2.applyColorMap(img,lut)

    return img_color
