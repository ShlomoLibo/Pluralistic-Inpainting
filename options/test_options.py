from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='./out', help='saves results here')
        parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--nsampling', type=int, default=1, help='ramplimg # times for each images')
        parser.add_argument('--save_number', type=int, default=10, help='choice # reasonable results based on the discriminator score')
        parser.add_argument('--grid', action="store_true", help='whether to output a grid of testing resutls')
        parser.add_argument('--num_display', type=int, default=5, help='if grid is True, number of images in each axis in the grid')

        self.isTrain = False

        return parser
