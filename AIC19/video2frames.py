import cv2
import os

split = 'train'
data_root = '/mnt/disk2/AIC19/aic19-track1-mtmc'
s_list = os.listdir(os.path.join(data_root, split))

for s_id in s_list:
    c_list = os.listdir(os.path.join(data_root, split, s_id))
    for c_id in c_list:
        # Read the video from specified path
        video_path = os.path.join(data_root, split, s_id, c_id, 'vdo.avi')
        cam = cv2.VideoCapture(video_path)

        # creating a folder named data
        images_path = os.path.join(data_root, split + '_images', c_id)
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        # frame
        currentframe = 1

        while True:
            # reading from frame
            ret, frame = cam.read()

            if ret:
                # if video is still left continue creating images
                name = os.path.join(images_path, '%06d.jpg' % currentframe)
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break

    # Release all space and windows once done
    # cam.release()
    # cv2.destroyAllWindows()
