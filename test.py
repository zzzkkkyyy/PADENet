import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data

import os, time, pdb, cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm

from synthetic import SyntheticDetection
from model import build_net

parser = argparse.ArgumentParser(description='PanoMonoDepth')

parser.add_argument('-b', '--batch_size', default=1)
parser.add_argument('--num_workers', default=0)
parser.add_argument('--ngpu', default=1)
parser.add_argument('--checkpoint', default='weights/checkpoint.pth')
parser.add_argument('--dataset_folder', default='datasets/')
parser.add_argument('--output_folder', default='outputs')

args = parser.parse_args()

def test(argv=None):
    label = args.dataset_folder.split('/')[-1]
    save_folder = "outputs_" + label + '_' + datetime.now().strftime('%m%d%H%M%S')
    os.makedirs(save_folder, exist_ok=True)

    net = build_net(ringpad=True)
    print("Load from checkpoint...")
    state_dict = torch.load(args.checkpoint)
    net.load_state_dict(state_dict)

    if args.ngpu > 0:
        net.cuda()
        cudnn.benchmark = True
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    print("Load dataset...")
    dataset = SyntheticDetection(root=args.dataset_folder, mode='val')
    batch_iterator = iter(data.DataLoader(dataset, 1, shuffle=False, num_workers=0))

    net.eval()
    for i in tqdm(range(len(dataset))):
        image = next(batch_iterator)
        if args.ngpu > 0:
            image = image.cuda()

        with torch.no_grad():
            disp = net(image)
            torch.cuda.empty_cache()

        disp_filename = "{0:0>6}".format(str(i))
        disp_pp = disp[-1].cpu().detach().numpy()[0, 0]
        np.save(os.path.join(save_folder, disp_filename), disp_pp)


if __name__ == '__main__':
    test()
