import cv2
import os

split = ''
data_root = '/mnt/disk2/AIC18/track1_videos'
image_root = '/mnt/disk2/AIC18/track1_images'
s_list = os.listdir(os.path.join(data_root, split))

for s_id in s_list:
    # Read the video from specified path
    video_path = os.path.join(data_root, split, s_id)
    cam = cv2.VideoCapture(video_path)

    # creating a folder named data
    images_path = os.path.join(image_root, s_id[:-4])
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
