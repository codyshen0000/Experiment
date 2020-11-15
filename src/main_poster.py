import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer


# args.model = 'RCAN'
# args.n_resgroups = 10
# args.n_resblocks = 20
# args.n_feats = 64
# args.chop = True
args.model = 'EEDSR'
args.n_resblocks = 32
args.n_feats = 256
args.res_scale = 0.1

args.save = 'EEDSR_poster_scratch_ncolor1_reducemean_temp'

args.scale = [4]
args.save_results = True
args.save_gt = True
args.patch_size = 128
# args.pre_train = "/home/lab532/Code/Shen/downscale/Pre-models/EDSR_x4.pt"#
args.data_range = '1-3000/3001-3018'
args.dir_data = '/home/lab532/Code/Shen/RCAN/RCAN_TrainCode/Datasets/ForEDSR/'
args.skip_threshold=1e3
args.n_threads = 6
args.batch_size = 16
# args.n_colors = 1

# args.test_only = True

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    # args.model = 'EDSR'
    # args.n_resblocks = 32
    # args.n_feats = 256
    # args.res_scale = 0.1

    # args.save = 'EDSR_poster

    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
