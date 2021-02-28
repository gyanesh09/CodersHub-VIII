import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import pickle
from sklearn.metrics import mean_squared_error

class Salary:
    def __init__(self):
        self.model_path = "./static/assets/SalaryPrediction.sav"
        if os.path.isfile(self.model_path) == True:
            #print("not trained")
            pass
        else:   
            dataset = pd.read_csv("C:/Users/asus/.spyder-py3/Salary_Data.csv")
            self.X=dataset.iloc[:, :-1].values
            self.Y=dataset.iloc[:, 1].values     
           
    def transform(self):
        X_train,X_test,Y_train,Y_test=train_test_split(self.X,self.Y,test_size=1/3,random_state=0)
        #print(X_)
        lr = LinearRegression()
        lr.fit(X_train,Y_train)
        pickle.dump(lr, open(self.model_path, 'wb')) 
        y_preds = lr.predict(X_test)
        print("Accuracy : ",mean_squared_error(y_preds,Y_test))     

    def predictor(self,value):
        if os.path.isfile(self.model_path) == True:
            print("transform not called")
            loaded_model = pickle.load(open(self.model_path, 'rb'))
            y_pred = loaded_model.predict(np.array(value).reshape(-1,1))
            print(y_pred[0])
            decplace=str(y_pred[0]).index('.')
            return str(y_pred[0])[:decplace+2] 

        else:
            #print("transform  called")
            self.transform()
            self.predictor(value)       


