import os
from shutil import copyfile

split = 'train'
data_root = '/mnt/disk2/AIC19/aic19-track1-mtmc'
s_list = os.listdir(os.path.join(data_root, split))

for s_id in s_list:
    c_list = os.listdir(os.path.join(data_root, split, s_id))
    for c_id in c_list:
        # Read the video from specified path
        gt_path = os.path.join(data_root, split, s_id, c_id, 'gt/gt.txt')
        gt_path_new = os.path.join(data_root, split + '_gt', c_id + '.txt')

        copyfile(gt_path, gt_path_new)
