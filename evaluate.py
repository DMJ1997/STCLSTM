import time

import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
from utils import *
from options.testopt import _get_test_opt
import nyudv2_dataloader
from resnet import resnet18
import modules
import net

args = _get_test_opt
TestImgLoader = nyudv2_dataloader.getTestingData_NYUDV2(args.batch_size, args.testlist_path, args.root_path)

Encoder = modules.E_resnet(resnet18)

if args.backbone in ['resnet50']:
    model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048], refinenet=args.refinenet)
elif args.backbone in ['resnet18', 'resnet34']:
    model = net.model(Encoder, num_features=512, block_channel=[64, 128, 256, 512], refinenet=args.refinenet)

model = nn.DataParallel(model).cuda()

if args.loadckpt is not None and args.loadckpt.endswith('.pth.tar'):
    print("loading the specific model in checkpoint_dir: {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict)
elif os.path.isdir(args.loadckpt):
    all_saved_ckpts = [ckpt for ckpt in os.listdir(args.loadckpt) if ckpt.endswith(".pth.tar")]
    print(all_saved_ckpts)
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    loadckpt = os.path.join(args.loadckpt, all_saved_ckpts[-1])
    start_epoch = int(all_saved_ckpts[-1].split('_')[-1].split('.')[0])
    print("loading the lastest model in checkpoint_dir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict)
else:
    print("You have not loaded any models.")


def test():
    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            print("Processing the {}th image!".format(batch_idx))
            image, depth = sample[0], sample[1]
            depth = depth.cuda()
            image = image.cuda()

            image = torch.autograd.Variable(image)
            depth = torch.autograd.Variable(depth)

            start = time.time()
            pred = model(image)
            end = time.time()
            running_time = end - start

            print(pred.size())
            print(depth.size())

            pred_ = np.squeeze(pred.data.cpu().numpy())
            depth_ = np.squeeze(depth.cpu().numpy())

            print(np.shape(pred_))
            print(np.shape(depth_))

            for seq_idx in range(len(pred_)):
                print(seq_idx)
                print(np.shape(depth_[0:]))

                depth = depth_[seq_idx]
                pred = pred_[seq_idx]

                d_min = min(np.min(depth), np.min(pred))
                d_max = max(np.max(depth), np.max(pred))
                depth = colored_depthmap(depth)
                pred = colored_depthmap(pred)

                print(d_min)
                print(d_max)

                filename = os.path.join('./samples/depth_' + str(seq_idx) + '.png')
                save_image(depth, filename)

                filename = os.path.join('./samples/pred_' + str(seq_idx) + '.png')
                save_image(pred, filename)


if __name__ == '__main__':
    test()