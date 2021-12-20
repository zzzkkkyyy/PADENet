import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data

import os, time, pdb
import argparse
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

from kitti import KittiDetection
from loss import MonodepthLoss
from model import build_net
from regularization import Regularization


parser = argparse.ArgumentParser(description='PanoMonoDepth')
parser.add_argument('--basenet', default='weights/vgg_pretrained.pth')
parser.add_argument('-b', '--batch_size', default=2)
parser.add_argument('--num_workers', default=0)
parser.add_argument('--ngpu', default=1)
parser.add_argument('--lr', '--learning_rate', default=1e-5)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--max_epoch', default=30)
parser.add_argument('--begin_epoch', default=-1)
parser.add_argument('--dataset_folder', default='datasets/')
parser.add_argument('--save_folder', default='weights/')
parser.add_argument('--log_folder', default='logs/')
parser.add_argument('--output_folder', default='outputs/')
args = parser.parse_args()


def vis_image(images):
    return (images[0] * 128. + torch.cuda.FloatTensor([104, 117, 123]).unsqueeze(-1).unsqueeze(-1))[[2, 1, 0], :, :] / 255

def train(argv=None):
    print('Load dataset...')
    dataset = KittiDetection(root=args.dataset_folder, mode='train', augment=True)
    dataset_val = KittiDetection(root=args.dataset_folder, mode='val', augment=False)
    batch_iterator = None

    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    net = build_net()

    if args.checkpoint == None:
        base_weights = torch.load(args.basenet)
        net.base.load_state_dict(base_weights)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                    elif 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0

        def xavier(param):
            init.xavier_uniform(param)

        print('Initialize params...')
        net.base.apply(weights_init)
        net.extra.apply(weights_init)
        net.upsample.apply(weights_init)

    else:
        print('Load from checkpoint...')
        state_dict = torch.load(args.checkpoint)
        net.load_state_dict(state_dict)

    if args.ngpu > 0:
        net.cuda()
        cudnn.benchmark = True

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    optimizer = optim.Adam(
        [
            {'params': net.base.parameters()},
            {'params': net.extra.parameters()},
            {'params': net.upsample.parameters()},
        ], lr=args.lr
    )

    regu = Regularization(net, 5e-4)

    mono_loss = MonodepthLoss()

    epoch_size = len(dataset) // args.batch_size
    epoch = int(args.begin_epoch)
    max_iter = (int(args.max_epoch) - epoch - 1) * epoch_size
    writer = SummaryWriter(logdir=args.log_folder)

    print('Begin running...')
    for iteration in range(max_iter):
        net.train()
        if iteration % epoch_size == 0:
            epoch += 1
            batch_iterator = iter(data.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers))

        images_left, images_right = next(batch_iterator)

        if args.ngpu > 0:
            images_left = images_left.cuda()
            images_right = images_right.cuda()

        left_disp = net(images_left)
        right_disp = net(images_right)

        optimizer.zero_grad()

        loss_disp, loss_rec, loss_smooth, loss_lr, right_est = mono_loss(left_disp, right_disp, images_left, images_right)
        loss_regu = regu()

        loss = loss_disp + loss_regu
        loss.backward()
        optimizer.step()

        if iteration % 200 == 0:
            print('Training ' + repr(epoch) + '/' + repr(iteration % epoch_size) + ' || Rec: ' + repr(loss_rec.item())
                + ' || Smooth: ' + repr(loss_smooth.item()) + ' || LR: ' + repr(loss_lr.item()) + ' || Regu: ' + repr(loss_regu.item()))

            writer.add_scalar('train_reconstruction_loss', loss_rec, iteration)
            writer.add_scalar('train_smooth_loss', loss_smooth, iteration)
            writer.add_scalar('train_lr_consistency_loss', loss_lr, iteration)
            writer.flush()

            if iteration % 2000 == 0:
                writer.add_image('train_input_left', vis_image(images_left), iteration)
                writer.add_image('train_input_right', vis_image(images_right), iteration)
                writer.add_image('train_disp_left', left_disp[-1][0] / left_disp[-1][0].max(), iteration)
                writer.flush()

        if iteration % 1000 == 0:
            net.eval()
            try:
                images_left, images_right = next(batch_iterator_val)
            except:
                batch_iterator_val = iter(data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=args.num_workers))
                images_left, images_right = next(batch_iterator_val)

            if args.ngpu > 0:
                images_left = images_left.cuda()
                images_right = images_right.cuda()

            with torch.no_grad():
                t0 = time.time()
                left_disp = net(images_left)
                right_disp = net(images_right)
                t1 = time.time()

                loss_disp, loss_rec, loss_smooth, loss_lr, right_est = mono_loss(left_disp, right_disp, images_left, images_right)
                loss_regu = regu()

                print('Validing ' + repr(epoch) + '/' + repr(iteration % epoch_size) + ' || Rec: ' + repr(loss_rec.item())
                    + ' || Smooth: ' + repr(loss_smooth.item()) + ' || LR: ' + repr(loss_lr.item()) + ' || Regu: ' + repr(loss_regu.item()))

                if iteration % 10000 == 0:
                    writer.add_scalar('valid_reconstruction_loss', loss_rec, iteration)
                    writer.add_scalar('valid_smooth_loss', loss_smooth, iteration)
                    writer.add_scalar('valid_lr_consistency_loss', loss_lr, iteration)
                    writer.add_image('valid_input_left', vis_image(images_left), iteration)
                    writer.add_image('valid_input_right', vis_image(images_right), iteration)
                    writer.add_image('valid_disp_left', left_disp[-1][0] / left_disp[-1][0].max(), iteration)
                    writer.flush()

        if iteration != 0 and iteration % (2 * epoch_size) == 0:
            torch.save(net.state_dict(), args.save_folder + datetime.now().strftime('%m%d%H%M%S') + '_epoch_' + str(iteration / epoch_size) + '.pth')

    torch.save(net.state_dict(), args.save_folder + 'final.pth')


if __name__ == '__main__':
    train()
