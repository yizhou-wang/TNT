import os
from shutil import copyfile

split = ''
data_root = '/mnt/disk2/AIC18/'
s_list = [name for name in os.listdir(os.path.join(data_root, 'GT_MOT')) if os.path.isdir(os.path.join(data_root, 'GT_MOT', name))]

for s_id in s_list:
    # Read the video from specified path
    gt_path = os.path.join(data_root, 'GT_MOT', s_id, 'gt/gt.txt')
    gt_path_new = os.path.join(data_root, 'track1_gt', s_id + '.txt')

    copyfile(gt_path, gt_path_new)
