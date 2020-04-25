# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:30:59 2020

@author: JANAKI
"""
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from model import model_choose

def get_args():
    parser = argparse.ArgumentParser(description="To detect faces from live webcam feed, pass them to the model, and display the face with the estimated age label.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default="AlexNet",
                        help="Enter the model's name you want to use.")
    parser.add_argument("--depth", type=int, default=16,
                        help="Enter the depth of WideResNet.")
    parser.add_argument("--width", type=int, default=8,
                        help="Enter the width of WideResNet")
    parser.add_argument("--weight_file", type=str, default=None,
                        help="enter the path to the weight file")
    args = parser.parse_args()
    return args

# =============================================================================
# def video():
#     capture = yield_images()
#     try:
#         yield capture
#     finally:
#         capture.release()
# =============================================================================
        
def yield_images():
    # capture video
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, img = video_capture.read()
        yield img

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def main():
    args = get_args()
    model_name = args.model_name
    weight_file = args.weight_file
    depth = args.depth
    width = args.width

    if not weight_file:
        weight_file = 'alexnet.hdf5'
        
    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    model = model_choose(depth, width, model_name=model_name)
    #model = buildmodel(64, 16, 8)
    model.load_weights(weight_file)
    img_size = model.input.shape.as_list()[1]

    image_generator = yield_images()
    margin = 0.8
    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            ages = np.arange(0, 101).reshape(101, 1)
            if (model_name == "WideResNet"):
                predicted_ages = results[1].dot(ages).flatten()
            else:
                predicted_ages = results.dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                label = str(int(predicted_ages[i]))
                draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("Real time Age Estimation", img)
        key = cv2.waitKey(30)

        if key == 27:
            break


if __name__ == '__main__':
    main()