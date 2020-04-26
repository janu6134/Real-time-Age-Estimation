# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:29:16 2020

@author: JANAKI
"""
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Dense, Activation, Flatten, BatchNormalization, Input, Dropout
from keras.regularizers import l2
from keras.applications import ResNet50
from keras.models import Model
from wide_resnet import buildmodel

def model_choose(depth, width, model_name="ResNet50", weights=None):
    if model_name == "ResNet50":
        base = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
        prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax", name="pred_age")(base.output) #he_normal draws samples from normal distribution of weights for intialization of layers
        model = Model(inputs=base.input, outputs=prediction, name = "ResNet-50")
        return model
    
    elif model_name == "AlexNet":
        input_shape=(200,200,3)
        num_classes=101
        input_layer = Input(shape=input_shape)
        
        #First layer of model
        x = ZeroPadding2D(padding = (0, 0))(input_layer)
        x = Conv2D(filters = 96, kernel_size = (11,11), strides = (4,4), padding = "valid", 
                   kernel_regularizer = l2(0.0), kernel_initializer = "he_normal", name = "conv1")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = "valid", name = "maxpool1")(x)
    
        #Second layer of model
        x = Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), padding = "same", 
                   kernel_regularizer = l2(0.0), kernel_initializer = "he_normal", name = "conv2")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool2")(x)
    
        #Third layer of model
        x = Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), padding = "same", 
                   kernel_regularizer = l2(0.0), kernel_initializer = "he_normal", name = "conv3")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    
        #Fourth layer of model
        x = Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), padding = "same", 
                   kernel_regularizer = l2(0.0), kernel_initializer = "he_normal", name = "conv4")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    
        #Fifth layer of model
        x = Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), padding = "same", 
                   kernel_regularizer = l2(0.0), kernel_initializer = "he_normal", name = "conv5")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool3")(x)
    
        #Sixth layer of model
        x = Flatten()(x)
        #x = Dropout(0.5)
        x = Dense(units=4096)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        #Seventh layer of model
        #x = Dropout(0.5)
        x = Dense(units=4096)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        #Eighth layer of model
        x = Dense(units=num_classes)(x)
        x = BatchNormalization()(x)
        x = Activation("softmax")(x)
    
        if weights is not None:
            x.load_weights(weights)
        model = Model(input_layer, x, name="AlexNet_8")
        return model

    else:
        print("Wide ResNet model")
        return buildmodel(64, depth, width)


def main():
    model = model_choose("AlexNet")
    model.summary()

if __name__ == '__main__':
    main()