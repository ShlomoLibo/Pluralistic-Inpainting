import os
import sys
import torch
from torch import nn
from torch import optim
import torchvision
from .mbu.mask_models import E1, E2, D_A, Disc, D_B, Disc2, Disc3
from .mbu.mask_utils import save_model, load_model, CustomDataset
import argparse
import time

  
from options import test_options
from dataloader import data_loader
from model import create_model
from itertools import islice

import torch
import torch.utils.data as data
import torchvision
import torchvision.utils as vutils
from PIL import Image


def save_masks(args, e1, e2, d_a, d_b, iters):
    test_domA, _= get_test_imgs(args)
    exps = []

    original_img_dir = os.path.join(args.output_dir, "original_images")
        if not os.path.exists(original_img_dir):
            os.makedirs(original_img_dir)
                
    masks_dir = os.path.join(args.output_dir, "masks")
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)
                
    for i in range(args.num_display):
        with torch.no_grad():
            exps.append(test_domA[i].unsqueeze(0))
            vutils.save_image(exps[i],
                              '%s/image_%06d.png' % (original_img_dir, i),
                              normalize=True, nrow=1)

    for i in range(args.num_display):
        separate_A = e2(test_domA[i].unsqueeze(0))
        common_A = e1(test_domA[i].unsqueeze(0))
        with torch.no_grad():
            AA_encoding = torch.cat([common_A, separate_A], dim=1)
            AA_decoding, mask2 = d_b(AA_encoding, test_domA[j])
            mask2_t = mask2 > 0.2
            mask2_t = mask2_t.float()
            exps.append(mask2_t)
            vutils.save_image(mask2_t,
                  '%s/mask_%06d.png' % (args.out, i),
                  normalize=True, nrow=1)
            mask_with_filter = color.rgb2gray(io.imread(f'%s/mask_%06d.png' % (args.out, i)))
            print(mask_with_filter.shape)
            mask_with_filter = ndimage.filters.maximum_filter(mask_with_filter, size=(3,3))
            io.imsave('%s/mask_expanded_%06d.png' % (args.output_dir, i), mask_with_filter)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/masks_%06d.png' % (args.output_dir, iters),
                      normalize=True, nrow=args.num_display + 1)
                      
#######################################


def get_test_imgs(args):


    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.resize, args.resize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    domA_test = CustomDataset(os.path.join(args.root, 'testA.txt'), transform=transform)
    domB_test = CustomDataset(os.path.join(args.root, 'testB.txt'), transform=transform)

    domA_test_loader = torch.utils.data.DataLoader(domA_test, batch_size=64,
                                                   shuffle=False, num_workers=0)
    domB_test_loader = torch.utils.data.DataLoader(domB_test, batch_size=64,
                                                   shuffle=False, num_workers=0)

    for domA_img in domA_test_loader:
        if torch.cuda.is_available():
            domA_img = domA_img.cuda()
        domA_img = domA_img.view((-1, 3, args.resize, args.resize))
        domA_img = domA_img[:]
        break

    for domB_img in domB_test_loader:
        if torch.cuda.is_available():
            domB_img = domB_img.cuda()
        domB_img = domB_img.view((-1, 3, args.resize, args.resize))
        domB_img = domB_img[:]
        break

    return domA_img, domB_img
    
########################################

def generate_masks(args):
    if args.gpu > -1:
        torch.cuda.set_device(args.gpu)
    
    sep = 25
    resize = 128
    
    e1 = E1(sep, resize // 64)
    e2 = E2(sep, resize // 64)
    d_a = D_A(resize // 64)
    d_b = D_B(resize // 64)

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        d_a = d_a.cuda()
        d_b = d_b.cuda()

    if args.load != '':
        save_file = os.path.join(args.load_model_for_mask, 'checkpoint')
        _iter = load_model_for_eval(save_file, e1, e2, d_a, d_b)

    e1 = e1.eval()
    e2 = e2.eval()
    d_a = d_a.eval()
    d_b = d_b.eval()
    
    save_masks(args, e1, e2, d_a, d_b, _iter)

#####################################################

def remove_glasses(opt):
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()

    for i, data in enumerate(islice(dataset, opt.how_many)):
        model.set_input(data)
        model.test()


################
def run_pipeline(args, opt):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    generate_masks(args)
    remove_glasses(opt)
    
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img', default='')
    parser.add_argument('--glasses_img', default='')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--load_model_for_mask', default='')
    parser.add_argument('--num_display', type=int, default=6)

    args = parser.parse_args()
    opt = test_options.TestOptions().parse()
    
    run_pipeline(args, opt)
