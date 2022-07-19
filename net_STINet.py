import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *
import torch.nn.functional as F
from dbpn import Net as DBPN
import torch
import dgl
import dgl.nn as dglnn
from DCNv2.dcn_v2 import DCN_sep
import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class Net(nn.Module):
    def __init__(self, base_filter, feat, num_stages, n_resblock, scale_factor):
        super(Net, self).__init__()
        self.scale_factor=scale_factor
        kernel = 8
        stride = 4
        padding = 2

        modules_down = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(1)]
        modules_down.append(ConvBlock(feat, feat, kernel, stride, padding, activation='lrelu', norm=None))
        self.motion_down = nn.Sequential(*modules_down)

        self.motion_feat = ConvBlock(4, base_filter, 3, 1, 1, activation='lrelu', norm=None)

        motion_net = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(2)]
        motion_net.append(ConvBlock(base_filter, feat, 3, 1, 1, activation='lrelu', norm=None))
        self.motion = nn.Sequential(*motion_net)

        self.upsample_layer = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        interp_l=[DeconvBlock(feat*3, feat, kernel, stride, padding, activation='lrelu', norm=None)]
        interp_l.extend([
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None)])
        interp_hl_res = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(n_resblock)]
        interp_l.extend(interp_hl_res)
        interp_l.extend([
            ResnetBlock(feat , kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None)])
        interp_l.extend([ConvBlock(feat, feat, kernel, stride, padding, activation='lrelu', norm=None)])
        self.interp_l= nn.Sequential(*interp_l)

        interp_h= [
            ConvBlock(feat * 3,feat, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None)]
        interp_h.extend(interp_hl_res)
        interp_h.extend([
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None)])
        interp_h.extend([ConvBlock(feat, feat, 3, 1, 1, activation='lrelu', norm=None)])
        self.interp_h = nn.Sequential(*interp_h)

        self.h_down=ConvBlock(feat, feat, kernel, stride, padding, activation='lrelu', norm=None)
        self.h_up=DeconvBlock(feat, feat, kernel, stride, padding, activation='lrelu', norm=None)

        layersAtBOffset = []
        layersAtBOffset.append(nn.Conv2d(feat*2, feat, 3, 1, 1, bias=True))
        layersAtBOffset.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersAtBOffset.append(nn.Conv2d(feat, feat, 3, 1, 1, bias=True))
        self.layersAtBOffset = nn.Sequential(*layersAtBOffset)
        self.layersAtB = DCN_sep(feat, feat, 3, stride=1, padding=1, dilation=1, deformable_groups=8)

        self.offsetdown=ConvBlock(feat, feat, kernel, stride, padding, activation='lrelu', norm=None)

        layersCtBOffset = []
        layersCtBOffset.append(nn.Conv2d(feat*2, feat, 3, 1, 1, bias=True))
        layersCtBOffset.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersCtBOffset.append(nn.Conv2d(feat, feat, 3, 1, 1, bias=True))
        self.layersCtBOffset = nn.Sequential(*layersCtBOffset)
        self.layersCtB = DCN_sep(feat, feat, 3, stride=1, padding=1, dilation=1, deformable_groups=8)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv11AtB=nn.Conv2d(feat*2, feat, 1, 1, 0, bias=True)

        self.l_conv11AtB=nn.Conv2d(feat, feat, 1, 1, 0, bias=True)
        self.h_conv11AtB=nn.Conv2d(feat, feat, 1, 1, 0, bias=True)

        layersFusion = []
        layersFusion.append(nn.Conv2d(feat*3, feat*3, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(feat*3, feat*3, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(feat*3, feat*3, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(feat*3, feat, 1, 1, 0, bias=True))
        self.layersFusion = nn.Sequential(*layersFusion)

        self.reconstruction_h = ConvBlock(feat, 3, 3, 1, 1, activation=None, norm=None)
        self.reconstruction_l = ConvBlock(feat, 3, 3, 1, 1, activation=None, norm=None)
        self.upconv1 = nn.Conv2d(feat, feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(feat, feat * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(feat, feat, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(feat, 3, 3, 1, 1, bias=True)

        ####ALIGNMENT

        self.DBPN = DBPN(num_channels=3, base_filter=64,  feat = 64, num_stages=5, scale_factor=4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.h_down_gcn=ConvBlock(feat, feat, kernel, stride, padding, activation='lrelu', norm=None)

        self.SAG1 = dglnn.SAGEConv(
            in_feats=feat, out_feats=feat*4, aggregator_type='gcn')
        self.SAG2 = dglnn.SAGEConv(
            in_feats=feat*4, out_feats=feat*4, aggregator_type='gcn')
        self.SAG3 = dglnn.SAGEConv(
            in_feats=feat*4, out_feats=feat*4, aggregator_type='gcn')
        self.SAG4 = dglnn.SAGEConv(
            in_feats=feat*4, out_feats=feat, aggregator_type='gcn')

        self.h_blending=nn.Conv2d(feat, 3, 1, 1, 0, bias=True)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        checkpoint = torch.load("/weights/pretrained/s-sr.pth", map_location=lambda storage, loc: storage)
        self.DBPN.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

        self.freeze_model(self.DBPN)


    def freeze_model(self, model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input, input_f_flow, input_b_flow, train=True):

        lr_feat=[]
        hr_feat=[]
        B, T, C, lH, lW = input.size()

        for i in range(T):
            h_ini = self.DBPN(input[:,i])
            l_ini = self.motion_down(h_ini)
            hr_feat.append(h_ini)
            lr_feat.append(l_ini)

        lr_feat=torch.stack(lr_feat, dim=1)
        hr_feat=torch.stack(hr_feat, dim=1)

        input_f_flow=input_f_flow/2
        input_b_flow= input_b_flow/2

        l_int=[]
        h_int=[]
        l_int.append(lr_feat[:,0])
        h_int.append(hr_feat[:,0])
        #ST-FI
        for i in range(T-1):
            motion_feat = self.motion_feat(torch.cat((input_f_flow[:,i], input_b_flow[:,i]), 1))
            M = self.motion(motion_feat)
            Lt=self.interp_l(torch.cat((lr_feat[:,i], lr_feat[:,i+1], M), 1))
            Ht = self.interp_h(torch.cat((hr_feat[:,i], hr_feat[:,i+1], self.upsample_layer(M)), 1))
            l_int.append(Lt)
            h_int.append(Ht)
            l_int.append(lr_feat[:,i+1])
            h_int.append(hr_feat[:,i+1])

        l_feats = torch.stack(l_int, dim=1)
        h_feats = torch.stack(h_int, dim=1)

        B, T, C, hH, hW = h_feats.size()
        B, T, C, lH, lW = l_feats.size()

        l_rec=[]
        h_rec=[]
        for i in range(T):
            h_rec.append(self.reconstruction_h(h_feats[:,i]))
            l_rec.append(self.reconstruction_l(l_feats[:,i]))

        l_rec = torch.stack(l_rec, dim=1)
        h_rec = torch.stack(h_rec, dim=1)
        #ST-LR
        l_comparison = []
        h_comparison=[]
        for i in range(T):
            if i == 0:
                idx = [0, 0, 1]
            else:
                if i == T - 1:
                    idx = [T - 2, T - 1, T - 1]
                else:
                    idx = [i - 1, i, i + 1]
            l_fea0 = l_feats[:, idx[0], :, :, :].contiguous()
            l_fea1 = l_feats[:, idx[1], :, :, :].contiguous()
            l_fea2 = l_feats[:, idx[2], :, :, :].contiguous()
            h_fea0 = h_feats[:, idx[0], :, :, :].contiguous()
            h_fea1 = h_feats[:, idx[1], :, :, :].contiguous()
            h_fea2 = h_feats[:, idx[2], :, :, :].contiguous()
            h_rfea0 = self.h_down(h_fea0)
            h_rfea2 = self.h_down(h_fea2)
            l_AtBOffset = self.layersAtBOffset(torch.cat([l_fea0, l_fea1], dim=1))
            l_CtBOffset = self.layersCtBOffset(torch.cat([l_fea2, l_fea1], dim=1))
            h_AtBOffset = self.layersAtBOffset(torch.cat([h_fea0, h_fea1], dim=1))
            h_CtBOffset = self.layersCtBOffset(torch.cat([h_fea2, h_fea1], dim=1))
            h_AtBOffset=self.offsetdown(h_AtBOffset)
            h_CtBOffset=self.offsetdown(h_CtBOffset)
            AtBfinaloffset=self.conv11AtB(torch.cat([l_AtBOffset,h_AtBOffset], dim=1))
            CtBfinaloffset = self.conv11AtB(torch.cat([l_CtBOffset, h_CtBOffset], dim=1))
            l_AtBfinaloffset=self.l_conv11AtB(AtBfinaloffset)
            l_CtBfinaloffset=self.l_conv11AtB(CtBfinaloffset)
            h_AtBfinaloffset=self.h_conv11AtB(AtBfinaloffset)
            h_CtBfinaloffset=self.h_conv11AtB(CtBfinaloffset)
            l_fea0_aligned = self.lrelu(self.layersAtB(l_fea0, l_AtBfinaloffset))
            l_fea2_aligned = self.lrelu(self.layersCtB(l_fea2,l_CtBfinaloffset))
            h_fea0_aligned = self.lrelu(self.layersAtB(h_rfea0, h_AtBfinaloffset))
            h_fea2_aligned = self.lrelu(self.layersCtB(h_rfea2,h_CtBfinaloffset))
            h_fea0_aligned=self.h_up(h_fea0_aligned)
            h_fea2_aligned=self.h_up(h_fea2_aligned)
            l_comparison.append(self.layersFusion(torch.cat([l_fea0_aligned, l_fea1, l_fea2_aligned], dim=1)))
            h_comparison.append(self.layersFusion(torch.cat([h_fea0_aligned, h_fea1, h_fea2_aligned], dim=1)))

        l_after_comparison = torch.stack(l_comparison, dim = 1)
        rl_feats = l_feats + l_after_comparison.view(B, T, C, lH, lW)
        h_after_comparison = torch.stack(h_comparison, dim = 1)
        rh_feats = h_feats + h_after_comparison.view(B, T, C, hH, hW)

        #ST-GR
        H_pool = []
        L_pool = []
        H_flatten=[]
        L_flatten=[]
        for i in range(T):
            H_pool.append(self.avg_pool(rh_feats[:, i, :, :, :].contiguous()).view(B, C))
            L_pool.append(self.avg_pool(rl_feats[:, i, :, :, :].contiguous()).view(B, C))
            L_flatten.append(rl_feats[:, i, :, :, :].view(B, C*lH*lW))
            H_flatten.append(self.h_down_gcn(rh_feats[:, i, :, :, :]).view(B, C*lH*lW))
        H_aftpool = torch.stack(H_pool, dim=1)
        L_aftpool = torch.stack(L_pool, dim=1)
        NX = torch.cat((H_aftpool, L_aftpool), dim=1)
        H_aftflatten = torch.stack(H_flatten, dim=1)
        L_aftflatten = torch.stack(L_flatten, dim=1)
        feaarr = torch.cat((H_aftflatten, L_aftflatten), dim=1)
        index1=[0, 0, 0, 0,0,0, 1, 1,1,1,1, 2,2,2,2, 3, 3,3, 4,4,5, 7,7,7, 7, 7, 7,
                8,8,8,8,8,9,9,9,9,10,10,10,11,11,12,0,1,2,3,4,5,6]
        index2=[1, 2, 3, 4,5,6, 2, 3,4,5,6, 3,4, 5,6, 4, 5,6, 5,6,6,8,9,10,11,12,13,
                9,10,11,12,13,10,11,12,13,11,12,13,12,13,13,7,8,9,10,11,12,13]
        u, v = torch.tensor(index1),torch.tensor(index2)
        bg = dgl.graph((u, v))
        bg = bg.to('cuda')
        seg_total=[]
        for i in range(B):
            bg.ndata['x'] = NX[i]
            sim=[]
            for j in range(len(index1)):
                sim.append([torch.cosine_similarity(NX[i][index1[j]],NX[i][index2[j]], dim=0),
                            torch.cosine_similarity(feaarr[i][index1[j]],feaarr[i][index2[j]], dim=0),
                            1-sigmoid(abs(index1[j]-index2[j]))])

            bg.edata['x'] = torch.tensor(sim, dtype=torch.float, device='cuda')
            seg1 = self.SAG1(bg, NX[i])
            seg1 = F.relu(seg1)
            seg2 = self.SAG2(bg, seg1)
            seg2 = F.relu(seg2)
            seg3 = self.SAG3(bg, seg2)
            seg3 = F.relu(seg3)
            seg4 = self.SAG4(bg, seg3)
            seg_total.append(seg4)
        seg_total=torch.stack(seg_total, dim=0)
        gh_arr=[]
        gl_arr=[]
        for i in range(7):
            gfeat1=seg_total[:,i].view(B, C, 1, 1)
            gfeat2= 0.1 * rh_feats[:,i,:] * gfeat1.expand_as(rh_feats[:,i,:]) + rh_feats[:,i,:]
            gh_arr.append(gfeat2)
            lfeat1=seg_total[:,i+7].view(B, C, 1, 1)
            lfeat2= 0.1 * rl_feats[:,i,:] * lfeat1.expand_as(rl_feats[:,i,:]) + rl_feats[:,i,:]
            gl_arr.append(lfeat2)
        gh_arr= torch.stack(gh_arr, dim=1)
        gl_arr = torch.stack(gl_arr, dim=1)
        rl_feats1 = gl_arr.view(B*T, C, lH, lW)

        r_out1 = self.lrelu(self.pixel_shuffle(self.upconv1(rl_feats1)))
        r_out1 = self.lrelu(self.pixel_shuffle(self.upconv2(r_out1)))

        r_out1 = self.lrelu(self.HRconv(r_out1))
        r_out1 = self.conv_last(r_out1)
        r_out1= r_out1.view(B, T, -1, hH, hW)
        h_rec2=[]
        for i in range(T):
            h_rec2.append(r_out1[:, i, :, :, :]+ self.h_blending(gh_arr[:, i, :, :, :]))

        h_rec2 = torch.stack(h_rec2, dim=1)
        h_finalrec=[]
        h_finalrec.append(h_rec)
        h_finalrec.append(h_rec2)
        h_finalrec=torch.stack(h_finalrec, dim=1)

        return l_rec,h_finalrec

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=35):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)