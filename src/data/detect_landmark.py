# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# https://github.com/facebookresearch/av_hubert/blob/main/avhubert/preparation/detect_landmark.py

import math
import os
import pickle
import sys
import time

import cv2
import dlib
import numpy as np
import skvideo
from tqdm import tqdm

from data.align_mouth import (
    apply_transform,
    crop_patch,
    cut_patch,
    landmarks_interpolate,
    write_video_ffmpeg,
)


def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for _, rect in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def preprocess_frames(base_path, face_predictor_path, mean_face_path):
    frame_path = os.path.join(base_path, "frames")
    lip_path = os.path.join(base_path, "lip")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]

    sorted_frames_files = sorted(
        os.listdir(frame_path), key=lambda x: int(x.split(".")[0].split("_")[1])
    )
    sorted_frames_files = [os.path.join(frame_path, f) for f in sorted_frames_files]

    print(f"preprocessing {len(sorted_frames_files)} frames")
    landmarks = []
    frames = [cv2.imread(frame_path) for frame_path in sorted_frames_files]
    # extract landmarks
    for frame in tqdm(frames):
        landmark = detect_landmark(frame, detector, predictor)
        landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)

    rois = crop_patch(
        frames,
        preprocessed_landmarks,
        lip_path,
        mean_face_landmarks,
        stablePntsIDs,
        STD_SIZE,
        window_margin=12,
        start_idx=48,
        stop_idx=68,
        crop_height=96,
        crop_width=96,
    )


def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames


def detect_face_landmarks(
    face_predictor_path, cnn_detector_path, root_dir, landmark_dir, flist_fn, rank, nshard
):
    def detect_landmark(image, detector, cnn_detector, predictor):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        if len(rects) == 0:
            rects = cnn_detector(gray)
            rects = [d.rect for d in rects]
        coords = None
        for _, rect in enumerate(rects):
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    predictor = dlib.shape_predictor(face_predictor_path)
    input_dir = root_dir  #
    output_dir = landmark_dir  #
    fids = [ln.strip() for ln in open(flist_fn).readlines()]
    num_per_shard = math.ceil(len(fids) / nshard)
    start_id, end_id = num_per_shard * rank, num_per_shard * (rank + 1)
    fids = fids[start_id:end_id]
    print(f"{len(fids)} files")
    for fid in tqdm(fids):
        output_fn = os.path.join(output_dir, fid + ".pkl")
        video_path = os.path.join(input_dir, fid + ".mp4")
        frames = load_video(video_path)
        landmarks = []
        for frame in frames:
            landmark = detect_landmark(frame, detector, cnn_detector, predictor)
            landmarks.append(landmark)
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        pickle.dump(landmarks, open(output_fn, "wb"))
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="detecting facial landmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=str, help="root dir")
    parser.add_argument("--landmark", type=str, help="landmark dir")
    parser.add_argument("--manifest", type=str, help="a list of filenames")
    parser.add_argument(
        "--cnn_detector",
        type=str,
        help="path to cnn detector (download and unzip from: http://dlib.net/files/mmod_human_face_detector.dat.bz2)",
    )
    parser.add_argument(
        "--face_predictor",
        type=str,
        help="path to face predictor (download and unzip from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)",
    )
    parser.add_argument("--rank", type=int, help="rank id")
    parser.add_argument("--nshard", type=int, help="number of shards")
    parser.add_argument("--ffmpeg", type=str, help="ffmpeg path")
    args = parser.parse_args()
    import skvideo

    skvideo.setFFmpegPath(os.path.dirname(args.ffmpeg))
    print(skvideo.getFFmpegPath())
    import skvideo.io

    detect_face_landmarks(
        args.face_predictor,
        args.cnn_detector,
        args.root,
        args.landmark,
        args.manifest,
        args.rank,
        args.nshard,
    )
