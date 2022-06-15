import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from dataset import Vimeo90k
from torch.utils.data import DataLoader
from SoftSplatModel import SoftSplatBaseline
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
from validation import Vimeo_PSNR as validate
from ReconLoss import LaplacianLoss, CensusLoss


def train():
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='SoftSplatBaseline', help='experiment name')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for training')
    parser.add_argument('--lr_schedule', nargs='*', type=int, help='lr schedule - when to decay. in epoch numbers.')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='lr schedule - how much to decay')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--loss_type', choices=['L1', 'MSE', 'Laplacian', 'Census'], default='Laplacian', help='loss function to use')
    parser.add_argument('--resume', type=int, default=0, help='epoch # to start / resume from.')
    parser.add_argument('--data_path', type=str, default='/Vimeo90k', help='path to dataset (Vimeo90k)')
    parser.add_argument('--save_dir', type=str, default='./ckpt', help='path to save checkpoint weights')
    parser.add_argument('--log_dir', type=str, default='./logs', help='path to tensorboard log')
    parser.add_argument('--val_dir', type=str, default='./valid', help='path to save validation results')
    parser.add_argument('--valid_batch_size', type=int, default=16, help='batch size for validation')
    args = parser.parse_args()
    print(args)

    # paths
    save_path = f'{args.save_dir}/{args.exp_name}.pth'
    logs = f'{args.log_dir}/{args.exp_name}'
    valid_path = f'{args.val_dir}/{args.exp_name}'

    # model
    model = SoftSplatBaseline()
    model = nn.DataParallel(model).cuda()

    # optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # dataset
    train_data = Vimeo90k(args.data_path)
    test_data = Vimeo90k(args.data_path, is_train=False)
    ipe = len(train_data) // args.batch_size
    print('iterations per epoch:', ipe)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(test_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=2)

    # loss
    best = 0
    loss_fn = None
    if args.loss_type == 'L1':
        loss_fn = nn.L1Loss()
    elif args.loss_type == 'MSE':
        loss_fn = nn.MSELoss()
    elif args.loss_type == 'Laplacian':
        loss_fn = LaplacianLoss()
    elif args.loss_type == 'Census':
        loss_fn = CensusLoss()

    if not args.resume == 0:  # if resume training
        print('loading checkpoints...')
        ckpt = torch.load(save_path)
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        optimizer.param_groups[0]['lr'] = args.lr
        del ckpt
        print('load complete!')
    else:
        if os.path.exists(logs):
            shutil.rmtree(logs)
        if os.path.exists(valid_path):
            shutil.rmtree(valid_path)

    milestones = args.lr_schedule
    if args.lr_schedule is None:
        milestones = [args.n_epochs + 10]

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    scheduler.last_epoch = args.resume - 1

    # recording & tracking
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)
    writer = SummaryWriter(logs)
    prev_best = None

    print('start training.')
    for epoch in range(args.resume, args.n_epochs):
        epoch_train_loss = 0
        model.train()
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            input_frames, target_frame, target_t, _ = data
            input_frames = input_frames.cuda().float()
            target_frame = target_frame.cuda().float()
            target_t = target_t.cuda().float()

            # forwarding
            pred_frt = model(input_frames, target_t)
            total_loss = loss_fn(pred_frt, target_frame)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()

        epoch_train_loss /= ipe
        torch.cuda.empty_cache()
        valid_psnr, valid_loss, cur_val_path = validate(model, valid_loader, valid_path, epoch)
        torch.cuda.empty_cache()
        writer.add_scalar('PSNR', valid_psnr, epoch)
        writer.add_scalar(f'{args.loss_type}/Train', epoch_train_loss, epoch)
        writer.add_scalar(f'{args.loss_type}/Valid', valid_loss, epoch)
        if valid_psnr > best:
            best = valid_psnr
            ckpt = {'opt': optimizer.state_dict(), 'model': model.module.state_dict()}
            torch.save(ckpt, save_path)
            # remove previous best validation.
            if prev_best is not None:
                shutil.rmtree(prev_best)
            prev_best = cur_val_path
        else:
            if prev_best is not None:
                shutil.rmtree(cur_val_path)

    print('end of training.')
    print('final validation.')
    torch.cuda.empty_cache()
    valid_psnr, valid_loss, cur_val_path = validate(model, valid_loader, valid_path, args.n_epochs)
    torch.cuda.empty_cache()
    writer.add_scalar('PSNR', valid_psnr, args.n_epochs)
    if valid_psnr > best:
        best = valid_psnr
        ckpt = {'opt': optimizer.state_dict(), 'model': model.module.state_dict()}
        torch.save(ckpt, save_path)
        # remove previous best validation.
        if prev_best is not None:
            shutil.rmtree(prev_best)
    else:
        if prev_best is not None:
            shutil.rmtree(cur_val_path)
    print(f'Final model PSNR: {best.item()}')


if __name__ == '__main__':
    train()
