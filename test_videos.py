import os
from utils.video_processing import get_video_info

real_folder = "dataset/real"
fake_folder = "dataset/fake"

print("\nREAL VIDEOS\n")

for file in os.listdir(real_folder):

    if file.endswith(".mp4"):
        path = os.path.join(real_folder, file)
        get_video_info(path)

print("\nFAKE VIDEOS\n")

for file in os.listdir(fake_folder):

    if file.endswith(".mp4"):
        path = os.path.join(fake_folder, file)
        get_video_info(path)