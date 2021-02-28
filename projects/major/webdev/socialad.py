import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pickle
import os

class Socialad:
    def __init__(self):
        self.model_path = "./static/assets/socialad.sav"
        if os.path.isfile(self.model_path) == True:
            #print("not trained")
            pass
        else:   
            self.dataset = pd.read_csv('C:/Users/asus/.spyder-py3/Social_Network_Ads.csv')

    def transform(self):
        self.X = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:, -1].values  
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)    
        with open(self.model_path,"wb") as f:
            pickle.dump(classifier,f) 
            pickle.dump(sc,f) 

    def predictor(self,v1,v2):
        try:
            with open(self.model_path, "rb") as f:
                loaded_model = pickle.load(f)
                sc = pickle.load(f)
                res=loaded_model.predict(sc.transform([[v1,v2]]))
            return int(res)    
        except:
            self.transform()  
            self.predictor(v1,v2)  
        