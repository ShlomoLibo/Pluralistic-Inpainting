import os
import sys
import torch
from torch import nn
from torch import optim
import torchvision
from mbu.mask_models import E1, E2, D_A, Disc, D_B, Disc2
from mbu.mask_utils import save_model, load_model_for_eval_pretrained, CustomDataset, load_model_for_eval
import argparse
import time
from skimage import color, io
from scipy import ndimage, misc

  
from options import test_options
from dataloader import data_loader
from model import create_model
from itertools import islice

import torch
import torch.utils.data as data
import torchvision
import torchvision.utils as vutils


def save_masks(args, e1, e2, d_a, d_b, iters):
    test_domA, _, _= get_test_imgs(args, False)
    exps = []

    original_img_dir = os.path.join(args.output_dir, "original_images")
    if not os.path.exists(original_img_dir):
        os.makedirs(original_img_dir)
                
    masks_dir = os.path.join(args.output_dir, "masks")
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    exp_masks_dir = os.path.join(args.output_dir, "exp_masks")
    if not os.path.exists(exp_masks_dir):
        os.makedirs(exp_masks_dir)

                
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
            AA_decoding, mask2 = d_b(AA_encoding, test_domA[i])
            mask2_t = mask2 > 0.2
            mask2_t = mask2_t.float()
            exps.append(mask2_t)
            vutils.save_image(mask2_t,
                  '%s/mask_%06d.png' % (masks_dir, i),
                  normalize=True, nrow=1)
            mask_with_filter = color.rgb2gray(io.imread(f'%s/mask_%06d.png' % (masks_dir, i)))
            mask_with_filter = ndimage.filters.maximum_filter(mask_with_filter, size=(3,3))
            io.imsave('%s/mask_expanded_%06d.png' % ( exp_masks_dir, i), mask_with_filter)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/masks_%06d.png' % (args.output_dir, iters),
                      normalize=True, nrow=args.num_display)



def save_new_images(args, e1, e2, d_a, d_b, iters):
    test_orig, test_domA, test_domB = get_test_imgs(args, True)
    exps = []

    final_img_dir = os.path.join(args.output_dir, "final_image")
    if not os.path.exists(final_img_dir):
        os.makedirs(final_img_dir)
    
    for i in range(args.num_display):
        with torch.no_grad():
            if i == 0:
                filler = test_orig[i].unsqueeze(0).clone()
                exps.append(filler.fill_(0))
                
            exps.append(test_orig[i].unsqueeze(0))
            
    for i in range(args.num_display):
        separate_A = e2(test_domA[i].unsqueeze(0))
        common_A = e1(test_domA[i].unsqueeze(0))
        exps.append(test_domA[i].unsqueeze(0))
        for j in range(args.num_display):
            with torch.no_grad():
                common_B = e1(test_domB[j].unsqueeze(0))
                BA_encoding = torch.cat([common_B, separate_A], dim=1)
                BA_decoding, mask = d_b(BA_encoding, test_domB[j])
                exps.append(BA_decoding)
                vutils.save_image(BA_decoding,
                      '%s/final_%06d.png' % (final_img_dir, i*args.num_display+j),
                      normalize=True, nrow=1)
            
    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/finals_%06d.png' % (final_img_dir, iters),
                      normalize=True, nrow=args.num_display+1)


def get_test_imgs(args, readC):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.resize, args.resize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    #original image
    domA_test = CustomDataset(os.path.join(args.root, 'testA.txt'), transform=transform)
    #feature image
    domB_test = CustomDataset(os.path.join(args.root, 'testB.txt'), transform=transform)
    #after removal
    if readC:
        domC_test = CustomDataset(os.path.join(args.root, 'testC.txt'), transform=transform)
    domA_test_loader = torch.utils.data.DataLoader(domA_test, batch_size=64,
                                                   shuffle=False, num_workers=0)
    domB_test_loader = torch.utils.data.DataLoader(domB_test, batch_size=64,
                                                   shuffle=False, num_workers=0)
    if readC:
        domC_test_loader = torch.utils.data.DataLoader(domC_test, batch_size=64,
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
    if readC:
        for domC_img in domC_test_loader:
            if torch.cuda.is_available():
                domC_img = domC_img.cuda()
            domC_img = domC_img.view((-1, 3, args.resize, args.resize))
            domC_img = domC_img[:]
            break
    if readC:
        return domA_img, domB_img, domC_img
    return domA_img, domB_img, domB_img
    

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
    args.load = args.load_mask
    if args.load != '':
        save_file = os.path.join(args.load, 'checkpoint')
        _iter = load_model_for_eval(save_file, e1, e2, d_a, d_b)
    e1 = e1.eval()
    e2 = e2.eval()
    d_a = d_a.eval()
    d_b = d_b.eval()
    save_masks(args, e1, e2, d_a, d_b, _iter)


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
    testC_file = os.path.join(opt.root, 'testC.txt')
    with open(testC_file, "w") as f:
        for i in range(opt.num_display):
            img_path = os.path.join('results', f'image_{i:06}_out_0.png')
            if i==0:
                f.write(img_path)
            else:
                f.write("\n"+img_path)


def add_glasses(args):
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
    args.load = args.load_transfer
    if args.load != '':
        save_file = os.path.join(args.load, 'checkpoint')
        _iter = load_model_for_eval(save_file, e1, e2, d_a, d_b)
    e1 = e1.eval()
    e2 = e2.eval()
    d_a = d_a.eval()
    d_b = d_b.eval()
    save_new_images(args, e1, e2, d_a, d_b, _iter)


def run_pipeline(args, opt):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("generating masks")
    generate_masks(args)
    print("removing glasses")
    remove_glasses(opt)
    print("adding glasses")
    add_glasses(args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img', default='')
    parser.add_argument('--root', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--glasses_img', default='')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--load_mask', default='')
    parser.add_argument('--load_transfer', default='')
    parser.add_argument('--load', default='')
    parser.add_argument('--num_display', type=int, default=6)
    ##### pluralistic arguments
    parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
    parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test')
    parser.add_argument('--nsampling', type=int, default=50, help='ramplimg # times for each images')
    parser.add_argument('--save_number', type=int, default=10, help='choice # reasonable results based on the discriminator score')
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment.')
    parser.add_argument('--model', type=str, default='pluralistic', help='name of the model type. [pluralistic]')
    parser.add_argument('--mask_type', type=int, default=[0],
                  help='mask type, 0: center mask, 1:random regular mask, '
                            '2: random irregular mask. 3: external irregular mask. [0],[1,2],[1,2,3]')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are save here')
    parser.add_argument('--which_iter', type=str, default='latest', help='which iterations to load')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
    # data pattern define
    parser.add_argument('--img_file', type=str, default='/data/dataset/train', help='training and testing dataset (images)')
    parser.add_argument('--img_feature_file', type=str, default='/data/dataset/train_feature', help='training and testing dataset (features)')
    parser.add_argument('--mask_file', type=str, default='none', help='load test mask')
    parser.add_argument('--loadSize', type=int, default=[128, 128], help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=[128, 128], help='then crop to this size')
    parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the image for data augmentation')
    parser.add_argument('--no_rotation', action='store_true', help='if specified, do not rotation for data augmentation')
    parser.add_argument('--no_augment', action='store_true', help='if specified, do not augment the image for data augmentation')
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--nThreads', type=int, default=8, help='# threads for loading data')
    parser.add_argument('--no_shuffle', action='store_true',help='if true, takes images serial')
    # display parameter define
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=int, default=1, help='display id of the web')
    parser.add_argument('--display_port', type=int, default=8097, help='visidom port of the web display')
    parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visidom web panel')
    parser.add_argument('--output_scale', type=int, default=4, help='# of number of the output scale')
    parser.add_argument('--isTrain', type=bool, default=False)

    args = parser.parse_args()
    opt = args
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
        if len(opt.gpu_ids):
            torch.cuda.set_device(opt.gpu_ids[0])
    
    run_pipeline(args, opt)
    
