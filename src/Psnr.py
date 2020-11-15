from utility import calc_psnr
import torch
import imageio
import os
import glob
import numpy as np




# dir_file = '/home/lab532/Code/Shen/downscale/Logs/AETAD-qq/results-results-Set5/'
gt_dir = '/home/lab532/Code/Shen/RCAN/RCAN_TrainCode/Datasets/ForEDSR/benchmark/Set5/HR'
sr_dir = '/home/lab532/Code/Shen/RCAN/RCAN_TrainCode/Datasets/ForEDSR/benchmark/Set5/Bicx4'
ext ='*.png'
gt_names_file = sorted([names for names in glob.glob(os.path.join(gt_dir, ext))])
sr_names_file = sorted([names for names in glob.glob(os.path.join(sr_dir, ext))])
p =0
for i in range(len(gt_names_file)):
    print(i)
    gt = imageio.imread(gt_names_file[i])
    if gt.ndim < 3:
        gt = gt[:, :, np.newaxis]
    gt = torch.from_numpy(gt.transpose((2, 0, 1)))
    gt = gt.float()
    sr = imageio.imread(sr_names_file[i])
    if sr.ndim < 3:
        sr = sr[:, :, np.newaxis]
    sr = torch.from_numpy(sr.transpose((2, 0, 1)))
    sr = sr.float()
    p += calc_psnr(sr, gt, scale=2, rgb_range=255,)
print(p/len(gt_names_file))

