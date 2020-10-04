import numpy as np
import cv2
import os
import argparse
from shapely.geometry import Polygon
from copy import deepcopy
import pickle


def bbox_overlap_ratio(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def MOT_to_UA_Detrac(gt_file, seq_name, save_folder, img_size):
    data = np.loadtxt(gt_file, delimiter=',', dtype=np.int64)
    M = data[:, :9]
    max_fr = np.max(data[:, 0])
    uniq_ids = np.unique(data[:, 1])
    # print(uniq_ids)
    for n in range(len(uniq_ids)):
        index = np.where(data[:, 1] == uniq_ids[n])
        M[:, 1][index] = n + 1
    max_id = len(uniq_ids)
    X = np.zeros(max_fr * max_id)
    Y = np.zeros(max_fr * max_id)
    W = np.zeros(max_fr * max_id)
    H = np.zeros(max_fr * max_id)
    X[M[:, 0] + (M[:, 1] - 1) * max_fr - 1] = M[:, 2]
    Y[M[:, 0] + (M[:, 1] - 1) * max_fr - 1] = M[:, 3]
    W[M[:, 0] + (M[:, 1] - 1) * max_fr - 1] = M[:, 4]
    H[M[:, 0] + (M[:, 1] - 1) * max_fr - 1] = M[:, 5]
    X = np.transpose(X.reshape(max_id, max_fr))
    Y = np.transpose(Y.reshape(max_id, max_fr))
    W = np.transpose(W.reshape(max_id, max_fr))
    H = np.transpose(H.reshape(max_id, max_fr))

    gt_info = {}
    gt_info['X'] = X
    gt_info['Y'] = Y
    gt_info['W'] = W
    gt_info['H'] = H

    # visibility
    V = np.zeros_like(W)
    for n in range(W.shape[0]):
        index = np.where(W[n, :] != 0)[0]
        if len(index) <= 1:
            V[n, index] = 1
            continue
        bbox = np.zeros((len(index), 4))
        bbox[:, 0] = X[n, index].T
        bbox[:, 1] = Y[n, index].T
        bbox[:, 2] = W[n, index].T
        bbox[:, 3] = H[n, index].T
        overlap_ratio = np.zeros((len(bbox), len(bbox)))

        for i in range(len(bbox)):
            boxA = deepcopy(bbox[i, :])
            boxA[2] = bbox[i, 0] + bbox[i, 2]
            boxA[3] = bbox[i, 1] + bbox[i, 3]
            for j in range(len(bbox)):
                boxB = deepcopy(bbox[j, :])
                boxB[2] = bbox[j, 0] + bbox[j, 2]
                boxB[3] = bbox[j, 1] + bbox[j, 3]

                box1 = [[boxA[0], boxA[1]], [boxA[2], boxA[1]], [boxA[2], boxA[3]], [boxA[0], boxA[3]]]
                box2 = [[boxB[0], boxB[1]], [boxB[2], boxB[1]], [boxB[2], boxB[3]], [boxB[0], boxB[3]]]

                iou = bbox_overlap_ratio(box1, box2)
                overlap_ratio[i, j] = iou

        for k in range(len(index)):
            overlap_ratio[k, k] = 0

        max_overlap = np.max(overlap_ratio, axis=1)
        V[n, index] = 1 - max_overlap

    gt_info['V'] = V
    gt_info['img_size'] = img_size

    save_path = os.path.join(save_folder, seq_name + '.pkl')
    f = open(save_path, 'wb')
    pickle.dump(gt_info, f)


def crop_UA_Detrac(gt_path, seq_name, img_folder, img_ext, save_folder):
    f = open(gt_path, 'rb')
    gt_info = pickle.load(f)
    # print(gt_info)
    margin_scale = 0.15
    resize_size = 182
    X = gt_info['X']
    Y = gt_info['Y']
    W = gt_info['W']
    H = gt_info['H']
    img_list = []
    for img_file in os.listdir(img_folder):
        if img_file.endswith(img_ext):
            img_list.append(img_file)

    for m in range(len(img_list)):
        img_name = img_list[m]
        img_path = os.path.join(img_folder, img_name)
        img = np.array(cv2.imread(img_path))
        img_size = img.shape
        num_id = H.shape[1]
        if m > gt_info['H'].shape[0]:
            continue
        for k in range(num_id):
            if gt_info['H'][m, k] < 1:
                continue
            xmin = round(X[m, k])
            ymin = round(Y[m, k])
            xmax = round(X[m, k] + W[m, k] - 1)
            ymax = round(Y[m, k] + H[m, k] - 1)
            min_side = min(xmax - xmin, ymax - ymin)
            margin = min_side * margin_scale
            xmin = round(max(xmin - margin, 1))
            ymin = round(max(ymin - margin, 1))
            xmax = round(min(xmax + margin, img_size[1])) + 1
            ymax = round(min(ymax + margin, img_size[0])) + 1
            crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
            crop_img = cv2.resize(crop_img, (resize_size, resize_size))
            class_name = seq_name + '_' + str(k + 1).zfill(8)
            class_folder = os.path.join(save_folder, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            id_name = class_name + '_' + str(m + 1).zfill(8)
            save_path = os.path.join(class_folder, id_name + '.png')
            cv2.imwrite(save_path, crop_img)


def create_pair(dataset_dir, save_file, num_pair=None, n_fold=None):
    class_list = os.listdir(dataset_dir)
    class_num = len(class_list)
    instance_num = np.zeros((1, class_num))

    for n in range(class_num):
        temp_dir = os.path.join(dataset_dir, class_list[n])
        sub_list = []
        for file in os.listdir(temp_dir):
            if file.endswith('.png'):
                sub_list.append(file)
        instance_num[:, n] = len(sub_list)
    with open(save_file, 'w') as g:
        K = n_fold  # 10
        pair_num = num_pair  # 300
        g.write("%d %d\n" % (K, pair_num))
        for k in range(K):
            for n in range(pair_num):
                d = 0
                while d < 1:
                    temp_num = 0
                    while temp_num < 2:
                        rand_class = np.random.randint(class_num)
                        class_name = class_list[rand_class]
                        img_list = []
                        for file in os.listdir(os.path.join(dataset_dir, class_name)):
                            if file.endswith('.png'):
                                img_list.append(file)
                        temp_num = int(instance_num[:, rand_class])
                    choose_idx = np.random.permutation(int(instance_num[:, rand_class]))[:2]
                    temp_name1 = img_list[choose_idx[0]]
                    temp_name2 = img_list[choose_idx[1]]
                    idx1 = int(temp_name1[:-4].split('_')[-1])
                    idx2 = int(temp_name2[:-4].split('_')[-1])
                    d = abs(idx1 - idx2)
                g.write("%s %d %d\n" % (class_name, idx1, idx2))

            for n in range(pair_num):
                rand_class = np.random.permutation(class_num)[:2]
                class_name1 = class_list[rand_class[0]]
                class_name2 = class_list[rand_class[1]]
                choose_idx1 = np.random.permutation(int(instance_num[:, rand_class[0]]))[0]
                choose_idx2 = np.random.permutation(int(instance_num[:, rand_class[1]]))[0]
                img_list1 = os.listdir(os.path.join(dataset_dir, class_name1))
                img_list2 = os.listdir(os.path.join(dataset_dir, class_name2))
                temp_name1 = img_list1[choose_idx1]
                temp_name2 = img_list2[choose_idx2]
                idx1 = int(temp_name1[:-4].split('_')[-1])
                idx2 = int(temp_name2[:-4].split('_')[-1])
                g.write("%s %d %d\n" % (class_name1, idx1, idx2))


def argument_parser():
    parser = argparse.ArgumentParser(description="Preprocess")
    parser.add_argument("--gt_folder",
                        type=str,
                        default="E:\\Project\\TNT\\pre-process\\data\\annotations")
    parser.add_argument("--img_folder",
                        type=str,
                        default="E:\\Project\\TNT\\pre-process\\data\\sequences")
    parser.add_argument("--save_folder",
                        type=str,
                        default="E:\\Project\\TNT\\pre-process\\data\\save_pkl")
    parser.add_argument("--crop_folder",
                        type=str,
                        default="E:\\Project\\TNT\\pre-process\\data\\crop_img")
    parser.add_argument("--valid_pairs_folder",
                        type=str,
                        default="E:\\Project\\TNT\\pre-process\\data\\valid_pairs")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    assert os.path.isdir(args.gt_folder)
    assert os.path.isdir(args.img_folder)
    seq_list = os.listdir(args.gt_folder)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    if not os.path.exists(args.crop_folder):
        os.makedirs(args.crop_folder)
    if not os.path.exists(args.valid_pairs_folder):
        os.makedirs(args.valid_pairs_folder)

    # MOT_to_UA_Detrac
    print("1st_step: converting MOT to UA Detrac format")
    for i in range(len(seq_list)):
        name = seq_list[i]
        seq_folder = os.path.join(args.img_folder, name[:-4])
        name_list = os.listdir(seq_folder)
        img = cv2.imread(os.path.join(seq_folder, name_list[0]))
        h = img.shape[0]
        w = img.shape[1]
        img_size = [w, h]
        MOT_to_UA_Detrac(os.path.join(args.gt_folder, name), name[:-4], args.save_folder, img_size)
    print("1st_step Done.")

    # crop_UA_Detrac
    print("2nd_step: crop images")
    file_list = os.listdir(args.save_folder)
    for i in range(len(file_list)):
        name = file_list[i]
        gt_path = os.path.join(args.save_folder, name)
        seq_name = name[:-4]
        img_file_folder = os.path.join(args.img_folder, seq_name)
        img_ext = '.jpg'
        crop_UA_Detrac(gt_path, seq_name, img_file_folder, img_ext, args.crop_folder)
    print("2nd_step Done.")

    # create_pair
    print("3rd_step: create pairs")
    valid_file = os.path.join(args.valid_pairs_folder, 'pairs.txt')
    create_pair(args.crop_folder, valid_file, 10, 300)
    print("3rd_step Done.")


if __name__ == '__main__':
    main()
