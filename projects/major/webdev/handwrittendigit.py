import tensorflow as tf
print(tf.__version__)
from os import path, getcwd, chdir
from django.conf import settings
import PIL
import numpy 
from PIL import Image, ImageOps 
from matplotlib import pyplot as plt
import os
import pickle
from django.core.files.storage import FileSystemStorage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class Digit:
    def __init__(self):
        self.model_path = "./static/assets/handwritten.sav"

    def transform(self):
        mnist = tf.keras.datasets.mnist
        mnist = tf.keras.datasets.mnist
        path = "C:/Users/asus/projects/major/mnist.npz"
        (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
        x_train=x_train/ 255.0
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        history = model.fit(x_train,y_train,epochs=5)
        model.save("./static/assets/my_model")
        
    
    def prediction(self,filename):
        path=os.path.join('C:/Users/asus/projects/major/media',filename)
        print(path)
        testimgpath=path
        #print(testimgpath)
        pic=PIL.Image.open(testimgpath)
        pic=ImageOps.grayscale(pic)
        pic=pic.resize((28,28))
        npar=numpy.asarray(pic)
        npar=npar/255.0
        npar =npar.reshape((-1,28,28))
        try:
            loadedmodel=tf.keras.models.load_model("./static/assets/my_model")
        except:
            self.transform()    
        probab=loadedmodel.predict(npar)
        print(numpy.argmax(probab[0]))
        return numpy.argmax(probab[0])