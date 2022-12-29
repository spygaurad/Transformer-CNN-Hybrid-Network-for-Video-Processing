import cv2
import os

path = 'Datasets/'
files = os.listdir(path)
files = [x for x in files if x[-1]=="4"]
save_path = "Datasets/Driving Dataset/train_images/image"
save_path2 = "Datasets/Driving Dataset/test_images/image"

i = 0
j = 0


for file in files:
    video = cv2.VideoCapture(f'{path}{file}')
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0
    while frame_counter < total_frames:
        _, frame = video.read()
        frame_counter += 1
        frame = cv2.resize(frame, (1280, 720))
        if frame_counter%10 == 0:
            cv2.imwrite(f'{save_path2}_{j}.png', frame)
            j+=1
        else:
            cv2.imwrite(f'{save_path}_{i}.png', frame)
            i += 1
    