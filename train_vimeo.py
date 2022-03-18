import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from dataset import Vimeo90k
from torch.utils.data import DataLoader
from SoftSplatModel import SoftSplatBaseline
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
from validation import Vimeo_PSNR as validate
from torch.nn.functional import interpolate


def train():
    exp_name = 'SoftSplatBaseline'
    clear_all = True

    resume = 0  # 0 for fresh training, else resume from pretraining.
    n_epochs = 80
    lr = 1e-4
    batch_size = 8
    valid_batch_size = 16
    loss_type = 'L1'

    # paths
    save_path = f'./{exp_name}.pth'
    valid_path = f'./valid/{exp_name}'
    logs = f'./logs/{exp_name}'
    dataset = '/Vimeo90k'

    # model
    model = SoftSplatBaseline()
    model = nn.DataParallel(model).cuda()

    # optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # dataset
    train_data = Vimeo90k(dataset)
    test_data = Vimeo90k(dataset, is_train=False)
    ipe = len(train_data) // batch_size
    print('iterations per epoch:', ipe)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(test_data, batch_size=valid_batch_size, shuffle=False, num_workers=2)

    # loss
    best = 0
    loss_fn = None
    if loss_type == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_type == 'MSE':
        loss_fn = nn.MSELoss()

    if not resume == 0:  # if resume training
        print('loading checkpoints...')
        ckpt = torch.load(save_path)
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        optimizer.param_groups[0]['lr'] = lr
        del ckpt
        print('load complete!')
    elif clear_all:
        if os.path.exists(logs):
            shutil.rmtree(logs)
        if os.path.exists(valid_path):
            shutil.rmtree(valid_path)

    # recording & tracking
    os.makedirs(logs, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)
    writer = SummaryWriter(logs)
    prev_best = None

    print('start training.')
    for epoch in range(resume, n_epochs):
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

            if not torch.isfinite(total_loss):
                raise ValueError(f'Error in Loss value: total:{total_loss.item()}')

            epoch_train_loss += total_loss.item()

        epoch_train_loss /= ipe
        torch.cuda.empty_cache()
        valid_psnr, valid_loss, cur_val_path = validate(model, valid_loader, valid_path, epoch)
        torch.cuda.empty_cache()
        writer.add_scalar('PSNR', valid_psnr, epoch)
        writer.add_scalars(f'{loss_type}', {'Train': epoch_train_loss, 'Valid': valid_loss}, epoch)
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
    valid_psnr, valid_loss, cur_val_path = validate(model, valid_loader, valid_path, n_epochs)
    torch.cuda.empty_cache()
    writer.add_scalar('PSNR', valid_psnr, epoch)
    writer.add_scalars(f'{loss_type}', {'Train': epoch_train_loss, 'Valid': valid_loss}, n_epochs)
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
    print(best.item())


if __name__ == '__main__':
    train()
