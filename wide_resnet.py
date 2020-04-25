# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 02:55:01 2020

@author: JANAKI
"""

import logging
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)

def wide_basic(no_of_inputs, no_of_outputs, ch_axis, stride):
    '''First, we have conv2. For our understanding, assume that the count = 2. So, the first set of operations will be (BatchNorm, ReLU), Conv) x count stacked with shortcut connection.
    These are added. Then, the above repeats one more time: (BatchNorm, ReLU, Conv) x2 stacked with the initial Bn,ReLU set.
    Then, we move on to conv3, and so on.
    '''
    def repeat(initial_layer):
        stride_parameters = [[stride], [(1,1)]]
        for i, j in enumerate(stride_parameters):
            #only for the first iteration
            if i == 0:
                #this is only during the first iteration , count = 1
                if no_of_inputs!=no_of_outputs:
                    initial_layer = BatchNormalization(axis = ch_axis)(initial_layer)
                    initial_layer = Activation("relu")(initial_layer)
                    conv_layer = initial_layer
                else:
                    #this happens when count > 1
                    conv_layer = BatchNormalization(axis = ch_axis)(initial_layer)
                    conv_layer = Activation("relu")(conv_layer)
                
                #this is the conv operation which happens regardless of the count we are in
                conv_layer = Conv2D(no_of_outputs, kernel_size = (3,3), strides = j[0],
                                    padding = "same", kernel_initializer = "he_normal", kernel_regularizer=l2(0.0005),
                                    use_bias=False)(conv_layer)
                print("default conv layer visited")
            
                #the above cycle is done twice according to value of i and if no_of_inputs!=no_of_outputs or vice versa
            else:
                #this is for when count > 1, ie second iteration
                conv_layer = Activation("relu")(BatchNormalization(axis = ch_axis)(conv_layer))
                conv_layer = Conv2D(no_of_outputs, kernel_size = (3,3), strides = j[0], padding = "same",
                                    kernel_initializer = "he_normal", kernel_regularizer = l2(0.0005), 
                                    use_bias = False)(conv_layer) 
        
        #Time for shortcut connections to be built!
        #Here, if no_of_inputs!=no_of_outputs, a shortcut connection is built with a conv on the initial layer. Otherwise, the initial set of (Bn, ReLU) is assigned as the shortcut connection
        if no_of_inputs!=no_of_outputs:
            shortcut_connection = Conv2D(no_of_outputs, kernel_size = (1,1), strides=stride, padding="same", 
                                         kernel_initializer = "he_normal", kernel_regularizer = l2(0.0005), 
                                         use_bias = False, name="shortcut%d"%no_of_inputs)(initial_layer)
        else:
            shortcut_connection = initial_layer
        
        return add([conv_layer, shortcut_connection]) #stacking up the layer
    return repeat
        
def layer(ch_axis, n_input_plane, n_output_plane, count, stride):
    def f(net):
        net = wide_basic(n_input_plane, n_output_plane, ch_axis, stride)(net)
        for i in range(2, int(count + 1)):
            net = wide_basic(n_output_plane, n_output_plane, ch_axis, stride=(1, 1))(net)
        return net

    return f
        
def buildmodel(image_size, depth, k):
    
    if K.image_data_format() == "channels_first":
        logging.debug("image_dim_ordering = 'th'")
        ch_axis = 1
        input_shape = (3, image_size, image_size)
    else:
        logging.debug("image_dim_ordering = 'tf'")
        ch_axis = -1
        input_shape = (image_size, image_size, 3)
    
    assert ((depth - 4) % 6 == 0)
    #The number of times we have to repeat a conv block is equal to (depth - 4) / 6
    #This parameter will be applied below as 'count' while calling the function layer to build the wide residual blocks.
    n = (depth - 4) / 6
            
    #input layer (64, 64, 3)
    input_layer = Input(shape=input_shape)

    kernels = [16, 16*k, 32*k, 64*k]
        
    #conv1 layer (64, 64, 16)
    conv1 = Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), padding = "same", kernel_initializer = "he_normal",
                   kernel_regularizer = l2(0.0005), use_bias = False, name="Conv1")(input_layer)

    # Add wide residual blocks
    print("Adding wide residual blocks:")
    #wideblock = wide_basic()
    
    #Stacking the second conv layer on the first conv layer we created above. This is conv2 ()
    conv2 = layer(ch_axis, n_input_plane = kernels[0], n_output_plane = kernels[1], count = n, stride = (1, 1))(conv1)
    
    #conv3 layer ()
    conv3 = layer(ch_axis, n_input_plane = kernels[1], n_output_plane = kernels[2], count = n, stride = (2, 2))(conv2)

    #conv4 layer ()
    conv4 = layer(ch_axis, n_input_plane = kernels[2], n_output_plane = kernels[3], count = n, stride = (2, 2))(conv3)
    
    bn = BatchNormalization(axis = ch_axis)(conv4)
    relu = Activation("relu")(bn)

    #Preparing for classification
    avgpool = AveragePooling2D(pool_size = (8, 8), strides = (1, 1), padding = "same")(relu)
    flatten = Flatten()(avgpool)
    
    #classification layers
    pre_estimate = Dense(units = 2, kernel_initializer = "he_normal", use_bias = False, kernel_regularizer = l2(0.0005), activation="softmax")(flatten)
    predicted_age = Dense(units = 101, kernel_initializer = "he_normal", use_bias = False, kernel_regularizer = l2(0.0005), activation="softmax")(flatten)
    
    #building the model
    model = Model(inputs=input_layer, outputs=[pre_estimate, predicted_age])
    return model


def main():
    model = buildmodel(64, 16, 8)
    model.summary()


if __name__ == '__main__':
    main()
