import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import cv2 as cv
import pickle
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten
from tensorflow.keras.layers import Reshape, BatchNormalization , Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt

class HindiCharacterRecognition:
    def __init__(self):
        self.model_path = "./static/assets/model_weights.hdf5"
        self.autoencoder_model_path = "./static/assets/encoder_model_weights.hdf5"
        self.y = []
        data_path = './static/assets/Encoder_Data/'
        for category in os.listdir(data_path):
            for label in os.listdir(data_path+category+'/'):
                for feature in os.listdir(data_path+category+'/'+label+'/'):
                    self.y.append(category+'/'+label)
        self.labels = {'ण': 0,
                        'ब': 1,
                        'भ': 2,
                        'च': 3,
                        'छ': 4,
                        'ड': 5,
                        'द	': 6,
                        'ढ': 7,
                        'ध': 8,
                        'ग': 9,
                        'घ': 10,
                        'ज्ञ': 11,
                        'ह': 12,
                        'ज': 13,
                        'झ': 14,
                        'क': 15,
                        'ख': 16,
                        'ङ': 17,
                        'क्ष': 18,
                        'ल': 19,
                        'म': 20,
                        'श': 21,
                        'न': 22,
                        'प': 23,
                        'स': 24,
                        'ष': 25,
                        'फ': 26,
                        'र': 27,
                        'ट': 28,
                        'त': 29,
                        'ठ': 30,
                        'थ': 31,
                        'त्र': 32,
                        'व': 33,
                        'य': 34,
                        'ञ': 35,
                        '०': 36,
                        '१': 37,
                        '२': 38,
                        '३': 39,
                        '४': 40,
                        '५': 41,
                        '६': 42,
                        '७': 43,
                        '८': 44,
                        '९': 45,
                        'अ': 46,
                        'आ': 47,
                        'इ': 48,
                        'ई': 49,
                        'उ': 50,
                        'ऊ': 51,
                        'ए': 52,
                        'ऐ': 53,
                        'ओ': 54,
                        'औ': 55,
                        'अं': 56,
                        'अः': 57}
        if os.path.isfile(self.model_path) == True and os.path.isfile(self.autoencoder_model_path) == True:
            #print("not trained")
            pass
        else:
            self.train_datagen = ImageDataGenerator( rescale=1.0/255.0,
                                                rotation_range=20,
                                                zoom_range=0.15,
                                                width_shift_range=0.05,
                                                height_shift_range=0.05,
                                                shear_range=0.05,
                                                horizontal_flip=True,
                                                fill_mode="nearest",
                                                validation_split=0.20)
                    
            self.valid_datagen = ImageDataGenerator( rescale=1.0/ 255.0,
                                                rotation_range=20,
                                                zoom_range=0.15,
                                                width_shift_range=0.05,
                                                height_shift_range=0.05,
                                                shear_range=0.05,
                                                horizontal_flip=True,
                                                fill_mode="nearest",
                                                validation_split=0.20)

            self.train_gen = self.train_datagen.flow_from_directory(
                                            './static/assets/HCR_DATA/train_data/',
                                            target_size=(28,28),
                                            color_mode="grayscale",
                                            class_mode="categorical",
                                            batch_size=32,
                                            shuffle=True,
                                            seed=7,
                                            interpolation="nearest")

            self.valid_gen = self.valid_datagen.flow_from_directory(
                                            './static/assets/HCR_DATA/test_data/',
                                            target_size=(28,28),
                                            color_mode="grayscale",
                                            class_mode="categorical",
                                            batch_size=32,
                                            shuffle=True,
                                            seed=7,
                                            interpolation="nearest")   

    
    def load_date(self):
        X = []
        y = []
        category_count=0
        label_count =0 
        feature_count = 0
        data_path = './static/assets/Encoder_Data/'
        for category in os.listdir(data_path):
            category_count+=1
            print("category_count : ",category_count)
            for label in os.listdir(data_path+category+'/'):
                label_count+=1
                print("label_count : ",label_count)
                for feature in os.listdir(data_path+category+'/'+label+'/'):
                    feature_count+=1
                    print("feature_count : ",feature_count)
                    X.append(cv.imread(data_path+category+'/'+label+'/'+feature,cv.COLOR_BGR2GRAY)/255.0)
                    y.append(category+'/'+label)
        enc = OneHotEncoder()
        enc_df = pd.DataFrame(enc.fit_transform(np.array(y).reshape(-1, 1)).toarray(),dtype='int8')
        y_encoded = enc_df
        #y_encoded
        X_train,X_test,y_train,y_test = train_test_split(X,y_encoded,test_size=0.30,random_state=4)
        return (X_train,y_train),(X_test,y_test)


    def create_autoencoder_model(self):

        image_width  = 28
        image_height = 28
        

        input_img = Input(shape=(image_width, image_height, 1))  

        # You can experiment with the encoder layers, i.e. add or change them
        x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(input_img)
        x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)

        # We need this shape later in the decoder, so we save it into a variable.
        encoded_shape = K.int_shape(x)

        x = Flatten()(x)
        x = Dense(256)(x)
        encoded = Dense(128)(x)

        # Builing the encoder
        encoder = Model(input_img,encoded,name='encoder')

        # at this point the representation is 128-dimensional
        encoder.summary()

        # Input shape for decoder
        encoded_input = Input(shape=(128,))
        x = Dense(np.prod(encoded_shape[1:]))(encoded_input)
        x = Reshape((encoded_shape[1], encoded_shape[2], encoded_shape[3]))(x)
        x = Conv2DTranspose(128,(3, 3), activation='relu',strides=2, padding='same')(x)
        x = Conv2DTranspose(64,(3, 3), activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(1,(3, 3), activation='sigmoid', padding='same')(x)

        decoder = Model(encoded_input,x,name='decoder')
        decoder.summary()

        autoencoder = Model(input_img, decoder(encoder(input_img)),name="autoencoder")
        autoencoder.summary()

        autoencoder.compile(optimizer='RMSprop', loss='binary_crossentropy')
        return autoencoder

    def create_model(self):
        model = Sequential()
        # The first two layers with 32 filters of window size 3x3
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28,28,1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(58, activation='softmax'))

        model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['acc'])
        return model

    def transform(self):
        n_epochs     = 100
        batch_size   = 64

        autoencoder = self.create_autoencoder_model()

        (X_train,y_train),(X_test,y_test) = self.load_date()
        X_train = np.array(X_train).reshape(-1,28,28,1)
        X_test = np.array(X_test).reshape(-1,28,28,1)

        history = autoencoder.fit(X_train, X_train,
                epochs=n_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test, X_test))
        autoencoder.save_weights(self.autoencoder_model_path)
        model = self.create_model()
        history = model.fit_generator(self.train_gen,validation_data=self.valid_gen,epochs=n_epochs)
        model.save_weights(self.model_path)  

    def predictor(self,value):
        if os.path.isfile(self.model_path) == True and os.path.isfile(self.autoencoder_model_path) == True:
            autoencoder_model = self.create_autoencoder_model()
            autoencoder_model.load_weights(self.autoencoder_model_path) 
            loaded_model = self.create_model()
            loaded_model.load_weights(self.model_path)
            y_pred = autoencoder_model.predict(value.reshape(-1,28,28,1))
            y_pred = loaded_model.predict(y_pred)
            y_pred = y_pred.argmax()
            print(y_pred)
            prediction = ""
            for key ,value in self.labels.items():
                if value == y_pred:
                    print(key)
                    prediction = key 
                    break
            return prediction     

        else:
            self.transform()
            self.predictor(value)       


