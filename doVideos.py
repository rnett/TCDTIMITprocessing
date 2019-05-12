import faulthandler
import os
import sys
import time
from sys import argv
from typing import List

import cv2
import dlib
import gc
import h5py
import imageio
import imutils
import librosa
import math
import numpy as np
from PIL import Image
from imutils import face_utils

lip_size = (30, 75)


class VideoFile:
    def __init__(self, base: str, newbase: str, speaker: str, clip: str):

        if not base.endswith("/"):
            base = base + "/"

        if not newbase.endswith("/"):
            newbase = newbase + "/"

        self.base = base
        self.speaker = speaker
        self.clip = clip
        self.newbase = newbase

        self.clipName = self.clip.replace(".mp4", "")

        self.file = base + speaker + "/Clips/straightcam/" + clip

        self.newfile = newbase + speaker + "_" + self.clipName + ".hdf5"

    def __str__(self):
        return "VideoFile: speaker: " + self.speaker + ", clip: " + self.clip


audio_framerate = 22050


def processVideoFile(video: VideoFile, detector, predictor, badVideos):
    try:
        wave, _ = librosa.load(video.file, mono=True, sr=audio_framerate)

        with imageio.get_reader(video.file) as durationReader:

            duration = durationReader.get_meta_data()['duration']

        fps = math.ceil(75 / duration)

        with imageio.get_reader(video.file, fps=fps) as reader:
            data = np.zeros(shape=(75, lip_size[0], lip_size[1]),
                            dtype=np.float32)

            for i, d in enumerate(reader):

                if i >= 75:
                    break

                gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

                # detect faces in the grayscale image
                rects = detector(gray, 1)

                isset = False

                # loop over the face detections
                for (k, rect) in enumerate(rects):
                    # determine the facial landmarks for the face region, then
                    # convert the landmark (x, y)-coordinates to a NumPy array
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # loop over the face parts individually
                    for (name, (l, m)) \
                            in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        # clone the original image so we can draw on it, then
                        # display the name of the face part on the image
                        if name == 'mouth':
                            # extract the ROI of the face region as a
                            # separate image

                            (x, y, w, h) = cv2.boundingRect(
                                np.array([shape[l:m]]))
                            roi = gray[y:y + h, x:x + w]
                            roi = imutils.resize(roi, width=250,
                                                 inter=cv2.INTER_CUBIC)
                            # roi = np.resize(roi,(100,250))

                            roi = np.array(Image.fromarray(roi).resize(
                                (lip_size[1], lip_size[0]), Image.ANTIALIAS))
                            isset = True
                            break

                if not isset:
                    print("\nCould not find mouth for speaker", video.speaker,
                          "clip", video.clip)
                    print("Error loading video for", str(video))

                    del data
                    gc.collect()
                    badVideos.append(video)
                    return False

                data[i] = roi
        # print("Read", i, "frames")
        # print("Writing to", video.newfile)

        if i != 75:
            raise ValueError("Wrong frames value of " + str(i))

        h5f = h5py.File(video.newfile, 'w')
        h5f.create_dataset("video", data=data, compression="gzip")
        h5f.create_dataset("audio", data=wave, compression="gzip")
        h5f.close()

        del data
        gc.collect()
        return True
    except:
        badVideos.append(video)
        return False


def get_all_videos(base: str, newbase: str) -> List[VideoFile]:
    if not base.endswith("/"):
        base = base + "/"

    if not newbase.endswith("/"):
        newbase = newbase + "/"

    speakers = os.listdir(base)

    videos = []

    for speaker in speakers:

        if speaker in ["55F", "56M", "57M", "58F", "59F"]:
            continue

        clips = os.listdir(base + speaker + "/Clips/straightcam/")
        clips = [c for c in clips if c.endswith(".mp4")]

        videos.extend(
            [VideoFile(base, newbase, speaker, clip) for clip in clips])

    return videos


if __name__ == '__main__':
    # with open("errors_py.log", 'w') as f:
    faulthandler.enable()

    videos = get_all_videos(argv[1], argv[2])

    detector = dlib.get_frontal_face_detector()

    if len(argv) > 3:
        predictor_path = argv[3]
    else:
        predictor_path = "./shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(predictor_path):
        print('Landmark predictor not found!')

    predictor = dlib.shape_predictor(predictor_path)

    print(len(videos), "videos to process")

    # executor = concurrent.futures.ThreadPoolExecutor(32)

    badVideos = []
    done = 0
    limit = len(videos)

    runTime = 0

    for video in videos:
        start = time.time()
        processVideoFile(video, detector, predictor, badVideos)
        end = time.time()

        duration = end - start

        runTime = (runTime * done + duration) / (done + 1)

        # executor.submit(processVideoFile, video, detector, predictor,
        #                 badVideos)
        done += 1

        sys.stdout.write(
            '\rDone {}/{} ({} %)  ETA: {} minutes'
                .format(done, limit,
                        int(100 * done / limit),
                        int(runTime * (limit - done) / 60)))
        sys.stdout.flush()

    # sleep(5 * 60)

    # executor.shutdown(wait=True)

    print(len(badVideos), "failures")
    for v in badVideos:
        print(v.speaker, ":", v.clip)
