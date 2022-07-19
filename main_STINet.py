import argparse
from math import log10
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from net_STINet import Net as STINet,FeatureExtractor
from data_STINet import get_training_set
import utils
import time
import cv2
import math
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# Training settings
parser = argparse.ArgumentParser(description='PyTorch STINet')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=30, help='training batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch for continuing training')
parser.add_argument('--total_iters', type=int, default=500000, help='Total iterations')
parser.add_argument('--snapshots', type=int, default=3, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=10, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=4, type=int, help='number of gpu')
#use the data path
parser.add_argument('--data_dir', type=str, default='/dataset/vimeo_septuplet/sequences')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--pretrained_sr', default='STINet.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
cudnn.benchmark = True
print(opt)


def train(epoch):
    epoch_loss = 0
    model.train()
    feature_extractor.eval()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, target_l, input_f_flow, input_b_flow,target_flow= \
            batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

        if cuda:
            input=Variable(input).cuda(gpus_list[0])
            input_f_flow = Variable(input_f_flow).cuda(gpus_list[0]).float()
            input_b_flow = Variable(input_b_flow).cuda(gpus_list[0]).float()
            target = Variable(target).cuda(gpus_list[0])
            target_l = Variable(target_l).cuda(gpus_list[0])
            target_flow = Variable(target_flow).cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        pred_l,  pred_h= model(input, input_f_flow, input_b_flow)

        l_rec_l = l_mse_loss_calc(pred_l, target_l)
        l_rec_h= h_mse_loss_calc(pred_h, target)
        l_per= feat_loss_calc(pred_h[:,1], target)
        pred_flow=flow_cal(pred_h[:,1])
        pred_flow = torch.tensor(pred_flow).cuda()

        MCL_abs=MCL_abs_loss_calc(pred_flow,target_flow)
        MCL_rel=MCL_rel_loss_calc(pred_flow)

        loss = l_rec_l +l_rec_h+ 0.1*l_per+0.1*MCL_abs+0.1*MCL_rel
        t1 = time.time()

        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        print(
            "===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(
                epoch, iteration, len(training_data_loader), loss.data,(t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def l_mse_loss_calc(pred, target):
    l_mse= criterion(pred, target)
    return l_mse

def h_mse_loss_calc(pred, target):
    h_mse=0
    for i in range(pred.shape[1]):
        h_mse_term=criterion(pred[:,i], target)
        h_mse+=h_mse_term
    return h_mse

def feat_loss_calc(pred, target):
    l_per = 0
    for i in range(pred.shape[1]):
        sr_feature = feature_extractor(pred[:,i])
        hr_feature = feature_extractor(target[:,i])
        l_per_h= criterion(sr_feature, hr_feature.detach())
        l_per= l_per+ l_per_h
    return l_per

def MCL_abs_loss_calc(pred, target):
    abs_mse= criterion(pred, target)
    return abs_mse

def get_pred_flow(input):
    flowarr=[]
    for i in range(len(input)):
        for j in range(len(input)):
            prvs = cv2.cvtColor(np.asarray(input[i]), cv2.COLOR_BGR2GRAY)
            next = cv2.cvtColor(np.asarray(input[j]), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flowarr.append(flow)
    flowarr=np.array(flowarr)
    return flowarr

def flow_cal(pred):
    pred_flow_arr=[]
    for i in range(pred.shape[0]):
        imglist = []
        for j in range(pred.shape[1]):
            img= utils.denorm(pred[i][j].cpu().data, vgg=True)
            img= img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
            img= abs(img) * 255
            imglist.append(img)
        pred_flow= get_pred_flow(imglist)
        pred_flow_arr.append(pred_flow)
    pred_flow_arr=np.array(pred_flow_arr)
    return pred_flow_arr

def MCL_rel_loss_calc(pred):
    rel_loss= 0
    for i in range(pred.shape[0]):
        for j in range(5):
            for m in range(pred.shape[2]):
                for n in range(pred.shape[3]):
                    for c in range(pred.shape[4]):
                        direction=torch.sign(pred[i,8*j-6,m,n,c])
                        if direction*(pred[i,8*j-6,m,n,c]-pred[i,8*j-5,m,n,c])>torch.tensor(0).cuda():
                            rel_loss+=direction*(pred[i,8*j-6,m,n,c]-pred[i,8*j-5,m,n,c])/10
    return rel_loss

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder + "STINet_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.upscale_factor, opt.data_augmentation, opt.file_list)
train_size = int(math.ceil(len(train_set) / opt.batchSize))
total_epochs = int(math.ceil(opt.total_iters / train_size))
print('total_epoch',total_epochs)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True,
                                  drop_last=True)

print('===> Building model ')
model = STINet(base_filter=32, feat=64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.MSELoss()

##VGG
feature_extractor = FeatureExtractor(models.vgg19(pretrained=True), feature_layer=35)
feature_extractor = torch.nn.DataParallel(feature_extractor, device_ids=gpus_list)

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])
    feature_extractor = feature_extractor.cuda(gpus_list[0])

optimizer = optim.Adamax(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_epoch, total_epochs + 1):
    train(epoch)
    # learning rate is decayed by a factor of 4 for every 10^5 iterations
    if (epoch + 1) % (total_epochs/ 5) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 4.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
    if (epoch + 1) % (opt.snapshots) == 0:
        checkpoint(epoch)

checkpoint(total_epochs)