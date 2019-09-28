import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import numpy as np
import argparse
import tqdm
from PIL import Image
# Model
import CRAFT.craft_utils as craft_utils
import CRAFT.imgproc as imgproc
import CRAFT.file_utils as file_utils
from CRAFT.craft import CRAFT
from collections import OrderedDict
# Data loader
from loader import SynthTextLoader


def load_statedict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def hard_worst_loss(loss, groundtruth):
    forceground_idx = groundtruth.nonzero().permute(1, 0)[0]
    background_idx = (groundtruth == 0).nonzero().permute(1, 0)[0]

    force_loss = loss[forceground_idx]
    back_loss = loss[background_idx]

    n_back_samples = min(forceground_idx.size(0) * 3, background_idx.size(0))
    top_k_back_loss_idx = np.argsort(-back_loss.data.cpu().numpy())[:n_back_samples]

    back_loss = back_loss[top_k_back_loss_idx]

    return (force_loss.sum() + back_loss.sum()) / (force_loss.size(0) + back_loss.size(0))


def OHEMLoss(pd_char_masks, gt_char_masks, pd_aff_masks, gt_aff_masks):
    b, h, w = gt_char_masks.size()
    # Flatten tensor
    pd_char_masks = pd_char_masks.contiguous().view(b * h * w)
    gt_char_masks = gt_char_masks.contiguous().view(b * h * w)
    pd_aff_masks = pd_aff_masks.contiguous().view(b * h * w)
    gt_aff_masks = gt_aff_masks.contiguous().view(b * h * w)

    # MSE loss for per pixel
    char_loss = torch.nn.MSELoss(reduction='none')(pd_char_masks, gt_char_masks)
    aff_loss = torch.nn.MSELoss(reduction='none')(pd_aff_masks, gt_aff_masks)

    # Find the worst loss
    hard_loss_char = hard_worst_loss(char_loss, gt_char_masks)
    hard_loss_aff = hard_worst_loss(aff_loss, gt_aff_masks)
    return hard_loss_char + hard_loss_aff


class ModelFit:
    def __init__(self, model, optimizer, train_logger):
        self.model = model
        self.optimizer = optimizer
        self.train_logger = train_logger
    
    def train(self, data_loader, data_folder, save_img, save_char, save_aff,
              num_epochs, batch_size, shuffle):
        self.model.train()

        loader = data_loader(batch_size, shuffle, data_folder, save_img, save_char, save_aff)
        loader = loader.loader()

        n_iter = len(loader)

        for epoch in range(num_epochs):
            print('Train model with epoch ', epoch + 1)
            total_loss = 0
            for batch_idx, (images, char_masks, aff_masks) in enumerate(loader):
                images = images.cuda()
                char_masks = char_masks.cuda()
                char_masks = char_masks[:, 0, :, :]
                aff_masks = aff_masks.cuda()
                aff_masks = aff_masks[:, 0, :, :]

                # Clear gradients
                self.optimizer.zero_grad()

                # Outputs
                output, _ = self.model(images)
                # Predict map for character
                out_char = output[:, :, :, 0]
                # Predict map for aff
                out_aff = output[:, :, :, 1]

                # Loss
                loss = OHEMLoss(out_char, char_masks, out_aff, aff_masks)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

        torch.save(self.model.state_dict(), 'checkpoints/model.pth')
    
    def valid(self, data_loader, data_folder, save_img, save_char, save_aff):
        self.model.eval()
        loader = data_loader(1, False, data_folder, save_img, save_char, save_aff)
        loader = loader.loader()

        n_iter = len(loader)
        valid_pbar = tqdm.tqdm(enumerate(loader), total=n_iter)
        with torch.no_grad():
            for batch_idx, (images, char_masks, aff_masks) in valid_pbar:
                images = images.cuda()
                char_masks = char_masks.cuda()
                char_masks = char_masks[:, 0, :, :]
                aff_masks = aff_masks.cuda()
                aff_masks = aff_masks[:, 0, :, :]

                # Outputs
                output, _ = self.model(images)
                # Predict map for character
                out_char = output[:, :, :, 0]
                # Predict map for aff
                out_aff = output[:, :, :, 1]

                image = out_char.cpu()
                import torchvision.transforms as transforms
                import matplotlib.pyplot as plt
                show = transforms.ToPILImage()
                image = show(image)
                image.save('debug.png')


if __name__ == '__main__':
    data_folder = '/mnt/data/hades/source/ocr_rnd/datasets/SynthText'
    save_img = '/mnt/data/hades/source/sekiwa_rnd/dataset/synthtext/images'
    save_char = '/mnt/data/hades/source/sekiwa_rnd/dataset/synthtext/char_mask'
    save_aff = '/mnt/data/hades/source/sekiwa_rnd/dataset/synthtext/aff_mask'

    # Load pre-trained model
    model = CRAFT()
    print('Loading weights from checkpoint')

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--trained_model', default='CRAFT/craft_mlt_25k.pth', type=str, help='Pretrained model')
    args = parser.parse_args()

    # if args.cuda:
    #     model.load_state_dict(load_statedict(torch.load(args.trained_model)))
    # else:
    #     model.load_state_dict(load_statedict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        model = model.cuda()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 betas=(0.9, 0.98), eps=1e-9)

    # Train model
    trainer = ModelFit(model, optimizer, None)
    num_epochs = 20
    batch_size = 1
    shuffle = True
    trainer.train(SynthTextLoader, data_folder, save_img, save_char, save_aff, num_epochs, batch_size, shuffle)
    trainer.valid(SynthTextLoader, data_folder, save_img, save_char, save_aff)

