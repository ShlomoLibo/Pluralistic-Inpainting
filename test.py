from options import test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice
import torchvision.utils as vutils
import torch


def save_grid(opt, model, dataset):
    batch = None
    for data in dataset:
        batch = data
        break
    imgs_f = batch['img_feature'] * 2 - 1
    imgs_truth = batch['img'] * 2 - 1
    imgs_masked = batch['mask'] * imgs_truth
    imgs_mask = batch['mask']
    imgs_c = (1 - batch['mask']) * imgs_truth

    exps = []
    exps_mask = []

    for i in range(opt.num_display):
        with torch.no_grad():
            if i == 0:
                filler = imgs_f[i].unsqueeze(0).clone()
                exps.append(filler.fill_(0))
                exps_mask.append(filler.fill_(0))

            exps.append(imgs_f[i].unsqueeze(0))
            exps_mask.append(imgs_f[i].unsqueeze(0))

    for i in range(opt.num_display):
        exps.append(imgs_truth[i].unsqueeze(0))
        exps_mask.append(imgs_masked[i].unsqueeze(0))
        for j in range(opt.num_display):
            with torch.no_grad():
                # setting input manually
                model.img_truth = imgs_truth[i].unsqueeze(0).cuda()
                model.img_f = imgs_f[j].unsqueeze(0).cuda()
                model.img_m = imgs_masked[i].unsqueeze(0).cuda()
                model.img_c = imgs_c[i].unsqueeze(0).cuda()
                model.mask = imgs_mask[i].unsqueeze(0).cuda()

                # running the model
                model.forward()

                exps.append(model.img_out.cpu())
                exps_mask.append(model.img_out.cpu())

    with torch.no_grad():
        exps = torch.cat(exps, 0)
        exps_mask = torch.cat(exps_mask, 0)

    vutils.save_image(exps,
                      '%s/experiments_%s.png' % (opt.results_dir, str(opt.which_iter)),
                      normalize=True, nrow=opt.num_display + 1)
    vutils.save_image(exps_mask,
                      '%s/masked_%s.png' % (opt.results_dir, str(opt.which_iter)),
                      normalize=True, nrow=opt.num_display + 1)


if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)

    if opt.grid:
        save_grid(opt, model, dataset)
    else:
        for i, data in enumerate(islice(dataset, opt.how_many)):
            model.set_input(data)
            model.test()