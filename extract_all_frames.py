import os
from utils.frame_extractor import extract_frames

real_folder = "dataset/real"
fake_folder = "dataset/fake"

for file in os.listdir(real_folder):

    if file.endswith(".mp4"):

        video_path = os.path.join(real_folder, file)

        output_folder = os.path.join(
            "dataset",
            "frames",
            "real",
            file.replace(".mp4", "")
        )

        extract_frames(video_path, output_folder)

for file in os.listdir(fake_folder):

    if file.endswith(".mp4"):

        video_path = os.path.join(fake_folder, file)

        output_folder = os.path.join(
            "dataset",
            "frames",
            "fake",
            file.replace(".mp4", "")
        )

        extract_frames(video_path, output_folder)