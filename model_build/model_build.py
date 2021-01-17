import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Input,BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from glob import glob
from sklearn.model_selection import train_test_split

class model:
    def __init__(self):
        self.train_dir = r"C:\Users\Aziz Shaikh\Desktop\mywork\skin-disease-build\dataset\train"
        self.test_dir = r"C:\Users\Aziz Shaikh\Desktop\mywork\skin-disease-build\dataset\test"
        self.regularizer = None
        self.train_data_generator = None
        self.test_data_generator = None
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        self.model = None

    def import_dataset(self):
        self.train_data_generator = ImageDataGenerator(validation_split = 0.2,rescale=1./255,horizontal_flip=True)
        self.test_data_generator = ImageDataGenerator(rescale=1./255)
        self.train_generator = self.train_data_generator.\
            flow_from_directory(self.train_dir,batch_size = 32,class_mode = 'sparse',
                                target_size = (200,200),subset = 'training')
        self.validation_generator = self.train_data_generator.\
            flow_from_directory(self.train_dir,batch_size = 32,class_mode = 'sparse',
                                target_size = (200,200),subset = 'validation')


    def model_build(self):
        i = Input(shape=(200, 200, 3))
        x = Conv2D(16, (3, 3), activation='relu')(i)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2())(x)
        # x = Dropout(0.2)(x)
        x = Dense(23, activation='softmax')(x)

        self.model = Model(i,x)

    def compile_model(self,x,y):


    def train_model(self):


    def save_model(self):


    def predict(self):

    