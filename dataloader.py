# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:30:21 2020

@author: JANAKI
"""
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from keras.utils import Sequence, to_categorical
import random
import math
import Augmentor
from pathlib import Path

'''utk face dataset has about 23,000+ images. The appa-real dataset has images divided into folders 'train', 'valid' and 'test'. 
The 'valid' folder of images is used for validation, while 'train' is used for training.'''

def data_augment(image):
    print("Data Augmentation")
    aug = Augmentor.Pipeline()
    aug.random_erasing(probability=0.5, rectangle_area=0.2)
    aug.flip_left_right(probability=0.5)
    aug.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    aug.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    aug.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    aug.random_distortion(probability=0.6, grid_width=2, grid_height=2, magnitude=8)
    #print("transform")
    imageres = [Image.fromarray(image)]
    
    for op in aug.operations:
        rand = round(random.uniform(0, 1), 1)
        if rand <= op.probability:
            imageres = op.perform_operation(imageres)
            #print("returning image[0]")
    return imageres[0]

class get_trainingset(Sequence):
    def __init__(self, appareal_data, utk_data=None, batch_size=32, image_size=224):
        self.path_age = [] #final list of consolidated images from appareal and utk
        self._appa_data(appareal_data) #path to appa-real dataset

        if utk_data:
            self._utk_data(utk_data) #path to utk dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_image = len(self.path_age) #number of images in the final list
        self.index = np.random.permutation(self.num_image)

    def __len__(self):
        return self.num_image // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)
        indexes = self.index[idx * batch_size:(idx + 1) * batch_size]

        for i, id in enumerate(indexes): #to return tuples of image and age
            image_path, age = self.path_age[id]
            image = cv2.resize(cv2.imread(str(image_path)), (image_size, image_size))
            x[i] =  data_augment(image)
            age += math.floor(np.random.randn() * 2 + 0.5)
            y[i] = np.clip(age, 0, 100)

        return x, to_categorical(y, 101)

    def on_epoch_end(self):
        self.index = np.random.permutation(self.num_image)

    def _utk_data(self, utk_data):
        image_dir = Path(utk_data)

        for image_path in image_dir.glob("*.jpg"):
            image_name = image_path.name  # the image's filename is of the format [age]_[gender]_[race]_[date&time].jpg
            age = min(100, int(image_name.split("_")[0])) #we take out only the age from the filename
            if image_path.is_file():
                self.path_age.append([str(image_path), age]) #append it to main list of consolidated images 

    def _appa_data(self, appareal_data):
        appa_path = Path(appareal_data)
        appa_train_data = appa_path.joinpath("train")
        df = pd.read_csv(str(appa_path.joinpath("gt_avg_train.csv"))) #this csv file contains the number of ratings, average apparent age, standard deviation of apparent age and the real age for each image

        for i, row in df.iterrows():
            age = min(100, int(row.apparent_age_avg)) #if the age of any image is above 100, it is restricted to 100
            image_path = appa_train_data.joinpath(row.file_name + "_face.jpg")
            if image_path.is_file():
                self.path_age.append([str(image_path), age]) #append it to main list of consolidated images 

class get_validationset(Sequence): #preparing validation dataset from appa-real database.
    def __init__(self, appareal_data, batch_size=32, image_size=224):
        self.batch_size = batch_size
        self.image_size = image_size
        self.path_age = []
        self._appa_data(appareal_data)
        self.num_image = len(self.path_age)

    def __len__(self):
        length = self.num_image // self.batch_size
        return length

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)

        for i in range(batch_size):
            image_path, age = self.path_age[idx * batch_size + i]
            image = cv2.imread(str(image_path))
            x[i] = cv2.resize(image, (image_size, image_size))
            y[i] = age

        return x, to_categorical(y, 101)

    def _appa_data(self, appareal_data):
        appa_path = Path(appareal_data)
        appa_valid_data = appa_path.joinpath("valid")
        df = pd.read_csv(str(appa_path.joinpath("gt_avg_valid.csv"))) #csv file containing the per image summaries of images in the folder 'valid' 

        for i, row in df.iterrows():
            age = min(100, int(row.apparent_age_avg))
            image_path = appa_valid_data.joinpath(row.file_name + "_face.jpg")

            if image_path.is_file():
                self.path_age.append([str(image_path), age])