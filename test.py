from options import test_options
import PIL
PIL.PILLOW_VERSION = PIL.__version__
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice

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

    for i, data in enumerate(islice(dataset, opt.how_many)):
        model.set_input(data)
        model.test()
