# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:28:50 2020

@author: JANAKI
"""

import argparse
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from pathlib import Path
import numpy as np
from dataloader import get_trainingset, get_validationset
from model import model_choose

def get_args():
    parser = argparse.ArgumentParser(description="To train the chosen model for age estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default="ResNet50",
                        help="Enter the desired model name: AlexNet/ResNet50/WideResNet")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Enter the batch size.")
    parser.add_argument("--n_epoch", type=int, default=30,
                        help="Enter the number of epochs you want to train the model for.")
    parser.add_argument("--appareal_data", type=str, required=True,
                        help="Enter the path to the APPA-REAL dataset")
    parser.add_argument("--utk_data", type=str, default=None,
                        help="Enter the path to the UTK face dataset")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Enter the desired learning rate for training.")
    parser.add_argument("--checkpoints", type=str, default="checkpoints",
                        help="Enter the directory to which you want to save the final models.")
    args = parser.parse_args()
    return args


class Schedule:
    def __init__(self, n_epoch, lr):
        self.epochs = n_epoch
        self.lr_ = lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.lr_
        elif epoch_idx < self.epochs * 0.50:
            return self.lr_ * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.lr_ * 0.04
        return self.lr_ * 0.008


def main():
    args = get_args()
    model_name = args.model_name
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    appareal_data = args.appareal_data
    utk_data = args.utk_data
    lr1 = args.lr

    if model_name == "AlexNet":
        image_size = 200
    elif model_name == "ResNet50":
        image_size = 224
    elif model_name == "WideResNet":
        image_size = 64
        
    #get the training and validation set
    train_data = get_trainingset(appareal_data, utk_data=utk_data, batch_size=batch_size, image_size=image_size)
    valid_data = get_validationset(appareal_data, batch_size=batch_size, image_size=image_size)
    
    model = model_choose(model_name=model_name)
    #model = WideResNet(image_size, depth=16, k=8)()
    
    opt = SGD(lr=lr1, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    
    checkpoints = Path(__file__).resolve().parent.joinpath(args.checkpoints)
    checkpoints.mkdir(parents=True, exist_ok=True)
    callbacks = [LearningRateScheduler(schedule=Schedule(n_epoch, lr=lr1)),
                 ModelCheckpoint(str(checkpoints) + "/weights.{epoch:02d}-{val_loss:.3f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")]

    hist = model.fit_generator(generator=train_data,
                               epochs=n_epoch,
                               validation_data=valid_data,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(checkpoints.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()