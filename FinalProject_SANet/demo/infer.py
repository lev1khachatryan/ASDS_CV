import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import net
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, sa_module, content, style, alpha=1,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f, style_f=feat_extractor(vgg, content, style)
    Fccc = sa_module(content_f,content_f)

    if interpolation_weights:
        _, C, H, W = Fccc.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = sa_module(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        Fccc=Fccc[0:1]
    else:
        feat = sa_module(content_f, style_f)
    feat = feat * alpha + Fccc * (1 - alpha)
    
    return decoder(feat)
  
def feat_extractor(vgg, content, style):
    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

    norm.to(device)
    enc_1.to(device)
    enc_2.to(device)
    enc_4.to(device)
    enc_5.to(device)

    Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
    Content5_1 = enc_5(Content4_1)

    Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
    Style5_1 = enc_5(Style4_1)

    content_f=[Content4_1,Content5_1]
    style_f=[Style4_1,Style5_1]

    return content_f, style_f