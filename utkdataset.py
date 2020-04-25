# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:48:34 2020

@author: JANAKI
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import dlib

'''We are given a utkface dataset divided into three sub-folders. These images are in the wild and not cropped to the face. This script takes these images, detects 
the faces in those images, adds a margin, and crops them. Then, the formatted images are stored in a folder uniformly.'''

def get_args():
    parser = argparse.ArgumentParser(description = "To take images from utkface, crop and store uniformly in a folder",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, required = True,
                        help="Enter the path to the input directory")
    parser.add_argument("--output_dir", type=str, required = True,
                        help="Enter the path to the output directory")
    args = parser.parse_args()
    return args

def crop(image, x1, y1, x2, y2): 
    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        resimg = cv2.copyMakeBorder(image, - min(0, y1), max(y2 - image.shape[0], 0), -min(0, x1), max(x2 - image.shape[1], 0), cv2.BORDER_REPLICATE)
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)
    return resimg[y1:y2, x1:x2, :]

def main():
    args = get_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    margin = 0.4
    detector = dlib.get_frontal_face_detector() #frontal face detector will detect the faces in the images
    output_dir.mkdir(parents = True, exist_ok = True)

    for image in tqdm(input_dir.glob("*/*.jpg")):
        res_img = cv2.imread(str(image))
        detected_faces = detector(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), 1)

        if len(detected_faces) != 1:
            continue
        
        x1, y1, x2, y2, w, h = detected_faces[0].left(), detected_faces[0].top(), detected_faces[0].right() + 1, detected_faces[0].bottom() + 1, detected_faces[0].width(), detected_faces[0].height()
        x_1 = int(x1 - margin * w)
        y_1 = int(y1 - margin * h)
        x_2 = int(x2 + margin * w)
        y_2 = int(y2 + margin * h)
        cropped = crop(res_img, x_1, y_1, x_2, y_2)
        cv2.imwrite(str(output_dir.joinpath(image.name)), cropped)


if __name__ == '__main__':
    main()