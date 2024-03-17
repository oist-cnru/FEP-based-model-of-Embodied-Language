import argparse
import yaml

from utils import str2bool, DictAction

def get_parser(add_help=False):

    #region arguments yapf: disable
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--epoch', default=5000, help='select epoch for evaluation')
    parser.add_argument('--save_result', type=str2bool, default=False, help='if ture, the output of the model will be stored')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epochs', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--num_regressions', type=int, default=50, help='stop error regression in which epoch')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
    parser.add_argument('--device_ids', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--port', type=int, default=8097, help='the port of visdom for visualization')

    # visualize and debug
    parser.add_argument('--log_interval', type=int, default=1, help='the interval for printing messages (#iteration)')
    parser.add_argument('--plot_interval', type=int, default=1000, help='the interval for ploting models (#iteration)')
    parser.add_argument('--save_interval', type=int, default=100, help='the interval for storing models (#epoch)')
    parser.add_argument('--eval_interval', type=int, default=1000, help='the interval for evaluating models (#epoch)')
    parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
    # parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not ')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of worker per gpu for data loader')
    parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
    # parser.add_argument('--debug', action="store_true", help='less data, faster loading')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--seed', default=0, help='seed for model parameters')
    parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
    # parser.add_argument('--weights', default=None, help='the weights for network initialization')
    # parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
    parser.add_argument('--checkpoint_path', default=None, help='the path of checkpoint for initialization')

    # optim
    parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--beta', type=float, default=1e-6, help='weighting on KL to prior')
    # parser.add_argument('--w1', default='[1, 1]', help='weighting on KL at t0 in pvrnn')
    # parser.add_argument('--w', '-metaprior', default='[1e-3, 1e-2]', help='weighting on KL in pvrnn')
    parser.add_argument('--k', type=float, default=1e2, help='weighting for language loss')
    parser.add_argument('--lang_loss', type=str, default="mse", help='loss function used for language')
    # parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
    # parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    # parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
    parser.add_argument('--num_context_frames', type=float, default=0.0001, help='number of ground truth frames to pass in before feeding in own predictions')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--regression_lr', type=float, default=0.01, help='learning rate for error regression')
    # endregion yapf: enable
    parser.add_argument('--plotlevel', type=int, default=10, help='Plot level of details (0-10)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda device nr')
    parser.add_argument('--sample_start', type=int, default=0, help='testsample index')
    parser.add_argument('--sample_num', type=int, default=10, help='number of samples to test')
    parser.add_argument('--sample_num_select', type=int, default=-1, help='select number of samples to test, in the range of possible samples')
    parser.add_argument('--trainstates', default='[500,1000,2000,2500,3000, 4000, 5000, 7500, 9000]', help='trainstates to evaluate')
    parser.add_argument('--evalversion', type=int, default=-1, help='version of eval algorithm')
    parser.add_argument('--predict_lang', type=int, default=False, help='during eval predict language when provided with behaviour sequence')

    parser.add_argument('--rep', type=str, default="", help='foldername')

    return parser



def load_arg(argv=None):
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args(argv)
    if p.config is not None:
        # load config file
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.SafeLoader)

        # update parser from config file
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {k}', k)
                assert k in key

        parser.set_defaults(**default_arg)

    args = parser.parse_args(argv)

    return args
