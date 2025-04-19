import argparse
import os
import math
from functools import partial
import random

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
from Utils.psnr_ssim import calculate_psnr, calculate_ssim
from Utils.lldataset import AllWeather
from losses.losses import CombinedLoss
import models
import utils
from utils import make_coord
from torchvision.utils import save_image
import torch.nn.functional as F

torch.backends.cudnn.enabled = False
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'


def make_coord_and_cell(img, scale):
    scale = int(scale)
    h, w = img.shape[-2:]
    h, w = h * scale, w * scale
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    return coord.unsqueeze(0), cell.unsqueeze(0)


def batched_predict(model, inp, coord, scale, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :].contiguous(), scale.contiguous(), cell[:, ql: qr, :].contiguous())
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_metrics(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
                 verbose=False, scale_max=4, window_size=0):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        psnr_metric_fn = calculate_psnr
        ssim_metric_fn = calculate_ssim
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        psnr_metric_fn = partial(calculate_psnr, dataset='div2k', scale=scale)
        ssim_metric_fn = partial(calculate_ssim, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        psnr_metric_fn = partial(calculate_psnr, dataset='benchmark', scale=scale)
        ssim_metric_fn = partial(calculate_ssim, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    psnr_res = utils.Averager()
    ssim_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        gt = (batch['gt'] - gt_sub) / gt_div

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp)
        else:
            pred = batched_predict(model, inp, coord, scale, cell, eval_bsize)

        pred = pred * gt_div + gt_sub

        pred.clamp_(0, 1)
        gt = np.squeeze(batch['gt'], axis=0).cuda()

        psnr = psnr_metric_fn(pred, gt, crop_border=0, test_y_channel=False)
        ssim = ssim_metric_fn(pred, gt, crop_border=0, test_y_channel=False)

        psnr_res.add(psnr.item())
        ssim_res.add(ssim.item())

        if verbose:
            pbar.set_description(f'val PSNR: {psnr_res.item():.4f}, SSIM: {ssim_res.item():.4f}')

    return psnr_res.item(), ssim_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--train_path', required=True, help='Path to the training dataset')
    parser.add_argument('--test_path', required=True, help='Path to the testing dataset')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Prepare training dataset and loader
    train_dataset = AllWeather(args.train_path, train=True, size=256)
    train_loader = DataLoader(train_dataset, batch_size=config['train_dataset']['batch_size'],
                              num_workers=config['train_dataset']['num_workers'], pin_memory=True)

    # Prepare testing dataset and loader
    test_dataset = AllWeather(args.test_path, train=False, size=None)
    test_loader = DataLoader(test_dataset, batch_size=config['test_dataset']['batch_size'],
                             num_workers=config['test_dataset']['num_workers'], pin_memory=True)

    # 选择SwinIR或HAT作为骨干网络
    if config['model']['name'] == 'SwinIR':
        from models.swinir import SwinIR
        model = SwinIR(**config['model']['params']).cuda()
    elif config['model']['name'] == 'HAT':
        from models.hat import HAT
        model = HAT(**config['model']['params']).cuda()
    elif config['model']['name'] == 'continuous-gaussian':
        from models.gaussian import ContinuousGaussian
        encoder_spec = config['model'].get('encoder_spec', {})
        cnn_spec = config['model'].get('cnn_spec', {})
        fc_spec = config['model'].get('fc_spec', {})
        model = ContinuousGaussian(encoder_spec, cnn_spec, fc_spec).cuda()
    else:
        raise ValueError(f"Unsupported model: {config['model']['name']}")

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_psnr = 0
    num_epochs = config['train']['num_epochs']

    for epoch in range(num_epochs):
        # 每100个epoch学习率衰减0.5
        if epoch % 100 == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for haze_tensor, clear_tensor, _ in pbar:
            haze_tensor = haze_tensor.cuda()
            clear_tensor = clear_tensor.cuda()

            optimizer.zero_grad()
            output = model(haze_tensor)
            loss = criterion(output, clear_tensor)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'Loss': train_loss / (len(pbar))})

        # Evaluate on the test set
        psnr, ssim = eval_metrics(test_loader, model,
                                  data_norm=config.get('data_norm'),
                                  eval_type=config.get('eval_type'),
                                  eval_bsize=config.get('eval_bsize'),
                                  verbose=True)

        print(f'Epoch {epoch + 1}: Test PSNR = {psnr:.4f}, SSIM = {ssim:.4f}')

        # Save the best model based on PSNR
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': best_psnr
            }, 'best_psnr_ckpt.pth')
            print(f'Saved best model with PSNR = {best_psnr:.4f}')
    