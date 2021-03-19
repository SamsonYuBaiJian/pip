import cv2
import os


def convert_avi_to_frame(source_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)

    videos = [os.path.join(source_folder, i) for i in os.listdir(source_folder)]

    for video in videos:
        video_id = video.split('/')[-1].split('.')[0]
        video_folder = os.path.join(target_folder, video_id)
        os.makedirs(video_folder, exist_ok=True)
        video_cap = cv2.VideoCapture(video)
        success, image = video_cap.read()
        count = 0
        while success:
            # save frame as JPEG file
            cv2.imwrite(os.path.join(video_folder, "frame_{}.jpg".format(count)), image)    
            success, image = video_cap.read()
            count += 1

convert_avi_to_frame('/nfs/home3/acct2011_02/CLEVEREST/dataset/dataset/contact/videos', '/nfs/home3/acct2011_02/CLEVEREST/dataset/dataset/contact/frames')
