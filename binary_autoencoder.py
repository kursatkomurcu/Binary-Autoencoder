import os, sys, csv
from re import T
import numpy as np
from numpy import unique
import pandas as pd
import matplotlib.pyplot as plt
import random
import argparse
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import plot_model
import larq as lq


class Class:
    def __init__(self):
        self.step_ = 12

    def load_data(self):
        """"
        out_X = []
        self.data = np.loadtxt('/home/kursat/projects/autoencoder/data/data.csv',
                                delimiter=',', skiprows=0)
        for i in range(12, self.data.shape[0]-10, 100):
            out_X.append(self.data[i-12:i, :])
        self.data = np.array(out_X)
        print('Data:', self.data.shape)"""
        self.data = pd.read_csv('/home/kursat/projects/autoencoder/data/data.csv')
        self.data = self.data.fillna(0)
        print('Data:', self.data.shape)
        return self.data

    def AE(self, data):
        """
        used larq library for binary autoencoder
        you should edit your data because threse are some values that contains two point. Ex: 2.333333.5474354
        because of data, our loss is nan but you can use better data for fix it
        """
        n_inputs = data.shape[1] #shape[1] means our columns (parameters)
        encoder_input = Input(shape=(n_inputs,))
        x = lq.layers.QuantDense(n_inputs*2, activation='hard_tanh',
                                input_quantizer='ste_sign',
                                kernel_quantizer='ste_sign',
                                kernel_constraint='weight_clip',
                                use_bias=False)(encoder_input)
        x = lq.layers.QuantDense(n_inputs, activation='hard_tanh',
                                input_quantizer='ste_sign',
                                kernel_quantizer='ste_sign',
                                kernel_constraint='weight_clip',
                                use_bias=False)(x)
      
        bottleneck = lq.layers.QuantDense(n_inputs, activation='hard_tanh',
                                        input_quantizer='ste_sign',
                                        kernel_quantizer='ste_sign',
                                        kernel_constraint='weight_clip',
                                        use_bias=False)(x) #middle point

        decoder_input = lq.layers.QuantDense(n_inputs, activation='hard_tanh',
                                            input_quantizer='ste_sign',
                                            kernel_quantizer='ste_sign',
                                            kernel_constraint='weight_clip',
                                            use_bias=False)(bottleneck)
        y = lq.layers.QuantDense(n_inputs*2, activation='hard_tanh',
                                input_quantizer='ste_sign',
                                kernel_quantizer='ste_sign',
                                kernel_constraint='weight_clip',
                                use_bias=False)(decoder_input)
        decoder_output = lq.layers.QuantDense(n_inputs, activation='hard_tanh',
                                            input_quantizer='ste_sign',
                                            kernel_quantizer='ste_sign',
                                            kernel_constraint='weight_clip',
                                            use_bias=False)(y)
        
        autoencoder = Model(encoder_input, decoder_output)
        
        print('AE')
        print(autoencoder.summary(expand_nested=True, show_trainable=True))
        print(lq.models.summary(autoencoder))
        return autoencoder

    def fitModel(self, data, model):
        print('FIT')
        data = np.asarray(data).astype(np.float32)
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(data, data, epochs=100, batch_size=64, verbose=2)
        model.save('binary-autoencoder.h5')

F = Class()
data = F.load_data()
print(data)
model = F.AE(data)    
#F.fitModel(data, model)