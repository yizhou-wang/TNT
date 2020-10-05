# TNT
Environment setting is follows. <br />
Operating system: windows, need to change `from scipy.io import loadmat` in `train_cnn_trajectory_2d.py` to `h5py` in Linux. <br />
Python: 3.6 <br />
tensorflow: 1.12.0 <br />
cuda: 9.0 <br />
cudnn: 7.1.2 <br />
opencv: 3.4.2 <br />
Other packages: numpy, pickle, sklearn, scipy, matplotlib, PIL, shapely. <br />

# 2D Tracking Testing
1. Prepare the detection data. <br />
follow the format of MOT (https://motchallenge.net/). <br />
The frame index and object index are from 1 (not 0) for both tracking ground truth and video frames. <br />
2. Set your data and model paths correctly on the top of `AIC19/tracklet_utils_3c.py`. <br />
The model can be downloaded from https://drive.google.com/drive/folders/1UJHoCz1P9rINqjHJP7ozGRnW5_rX2IXl?usp=sharing. <br />
3. Set the `file_len` to be the string length of your input frame name before the extension. <br />
4. Adjust the tracking parameters in `track_struct['track_params']` of `AIC19/tracklet_utils_3c.py` in the function `TC_tracker()`. <br />
5. Run python `AIC19/TC_tracker.py`. <br />
6. Run `post_deep_match.py` for post processing if necessary.
7. The SCT results are saved in the `txt_result` folder.
8. Run `get_GPS.m` to obtain the GPS location of the tracking results.
9. The final outputs are saved in the `txt_GPS_new` folder.

# SCT results
The SCT results are saved in TNT/AIC19/txt_GPS_new.

# 2D Tracking Training
1. Prepare training data
```
python pre-process/preprocess.py --gt_folder <DATA_ROOT>/train_gt/ \
    --img_folder <DATA_ROOT>/train_images/ \
    --save_folder <DATA_ROOT>/train_prep/ \
    --crop_folder <DATA_ROOT>/train_crop/ \
    --valid_pairs_folder <DATA_ROOT>/train_valid_pairs
```

2. Validate data
```
python src/validate_on_lfw.py <DATA_ROOT>/train_crop/ \
    <DATA_ROOT>/models/triplet_model/AI_city_model/ \
    --lfw_pairs <DATA_ROOT>/train_valid_pairs/pairs.txt \
    --distance_metric 0 --use_flipped_images \
    --subtract_mean --use_fixed_image_standardization
```

3. Train FaceNet triplet loss
```
python src/train_tripletloss.py --logs_base_dir <DATA_ROOT>/models/aicity19_new/facenet/logs \
    --models_base_dir <DATA_ROOT>/models/aicity19_new/facenet/models \
    --data_dir <DATA_ROOT>/train_crop/ \
    --lfw_dir <DATA_ROOT>/train_crop/ \
    --lfw_pairs <DATA_ROOT>/train_valid_pairs/pairs.txt \
    --image_size 160 --model_def models.inception_resnet_v1 \
    --optimizer RMSPROP --weight_decay 1e-4 --max_nrof_epochs 500 \
    --embedding_size 512 --batch_size 30 --people_per_batch 15 \
    --images_per_person 10 --epoch_size 100 --learning_rate 0.0001
```

4. Set TNT 2D tracking configs <br>
Set directory paths in `train_cnn_trajectory_2d.py` before the definition of all the functions.
Change the sample probability `sample_prob` according to your data density.
The number of element in `sample_prob` is the number of your input `pkl` files.
Set the learning rate (lr) to 1e-3 at the beginning.
Every 2000 steps, decrease lr by 10 times until it reaches 1e-5.
The output model will be stored in save_dir.

5. Run training script
```
python train_cnn_trajectory_2d.py
```

# Citation
Use this bibtex to cite this repository:
```
@inproceedings{wang2019exploit,
  title={Exploit the connectivity: Multi-object tracking with trackletnet},
  author={Wang, Gaoang and Wang, Yizhou and Zhang, Haotian and Gu, Renshu and Hwang, Jenq-Neng},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={482--490},
  year={2019},
  organization={ACM}
}
```
