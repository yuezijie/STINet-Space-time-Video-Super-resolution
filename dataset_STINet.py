import torch.utils.data as data
import torch
import numpy as np
import os
from os.path import join
from PIL import Image, ImageOps
import random
import cv2

max_flow = 150.0
np.random.seed(1)

def load_img(filepath, scale):
    list = os.listdir(filepath)
    list.sort()
    h_random = int(np.random.uniform(0, 1) * 64)
    w_random = int(np.random.uniform(0, 2.5) * 64)
    target = [modcrop(Image.open(filepath + '/' + list[i]).convert('RGB'), h_random, w_random) for i in
              range(0,7)]
    h, w = target[0].size
    h_in, w_in = int(h // scale), int(w // scale)
    target_l = [lr.resize((h_in, w_in), Image.BICUBIC) for lr in target]
    input = [target_l[j] for j in [0,2,4,6]]
    return input, target, target_l, list

def modcrop(img, h_random, w_random):
    img = img.crop((w_random, h_random, w_random + 64, h_random + 64))
    return img

def augment(img_in, img_tar, img_tar_l, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    if random.random() < 0.5 and flip_h:
        img_in = [ImageOps.flip(j) for j in img_in]
        img_tar = [ImageOps.flip(j) for j in img_tar]
        img_tar_l = [ImageOps.flip(j) for j in img_tar_l]
        info_aug['flip_h'] = True
    if rot:
        if random.random() < 0.5:
            img_in = [ImageOps.mirror(j) for j in img_in]
            img_tar = [ImageOps.mirror(j) for j in img_tar]
            img_tar_l = [ImageOps.mirror(j) for j in img_tar_l]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = [j.rotate(180) for j in img_in]
            img_tar = [j.rotate(180) for j in img_tar]
            img_tar_l = [j.rotate(180) for j in img_tar_l]
            info_aug['trans'] = True
    return img_in, img_tar, img_tar_l, info_aug

def get_input_flow(input):
    f_flowarr=[]
    b_flowarr=[]
    for i in range(len(input)-1):
        prvs = cv2.cvtColor(np.asarray(input[i]), cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(np.asarray(input[i+1]), cv2.COLOR_BGR2GRAY)
        f_flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        b_flow = cv2.calcOpticalFlowFarneback(next, prvs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        f_flowarr.append(f_flow)
        b_flowarr.append(b_flow)
    f_flowarr=np.array(f_flowarr)
    b_flowarr=np.array(b_flowarr)
    return f_flowarr,b_flowarr

def get_target_flow(input):
    flowarr=[]
    for i in range(len(input)):
        for j in range(len(input)):
            prvs = cv2.cvtColor(np.asarray(input[i]), cv2.COLOR_BGR2GRAY)
            next = cv2.cvtColor(np.asarray(input[j]), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flowarr.append(flow)
    flowarr=np.array(flowarr)
    return flowarr

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, upscale_factor, data_augmentation, file_list, transform=None):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir, file_list))]
        self.image_filenames = [join(image_dir, x) for x in alist]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        input, target, target_l, file_list = load_img(self.image_filenames[index], self.upscale_factor)
        if self.data_augmentation:
            input, target, target_l, _ = augment(input, target, target_l)
        input_f_flow,input_b_flow=get_input_flow(input)
        target_flow=get_target_flow(target)
        if self.transform:
            input = [self.transform(j) for j in input]
            input = torch.tensor([item.detach().numpy() for item in input])
            target = [self.transform(j) for j in target]
            target = torch.tensor([item.detach().numpy() for item in target])
            target_l = [self.transform(j) for j in target_l]
            target_l= torch.tensor([item.detach().numpy() for item in target_l])
            input_f_flow = [torch.from_numpy(i.transpose(2, 0, 1)) for i in input_f_flow]
            input_f_flow= torch.tensor([item.detach().numpy() for item in input_f_flow])
            input_b_flow = [torch.from_numpy(i.transpose(2, 0, 1)) for i in input_b_flow]
            input_b_flow = torch.tensor([item.detach().numpy() for item in input_b_flow])
        return input, target, target_l, input_f_flow, input_b_flow, target_flow, \
               file_list, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)