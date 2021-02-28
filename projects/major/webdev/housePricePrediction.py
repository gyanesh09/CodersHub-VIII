from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os
from sklearn.metrics import mean_squared_error

class Model:
    def __init__(self):
        self.model_path = "./static/assets/housePredicitonModel.sav"
        if os.path.isfile(self.model_path) == True:
            #print("not trained")
            pass
        else:   
            #print("training req")
            self.X = np.arange(1,101,1)
            temp = 2*self.X
            temp = temp+1
            self.Y  =  temp
            
        #print(self.X)
        #print(self.Y)

    def transform(self):
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.Y,test_size=0.25,random_state=7)
        lr = LinearRegression()
        lr.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))
        pickle.dump(lr, open(self.model_path, 'wb'))
        y_preds = lr.predict(X_test.reshape(-1,1))
        print("Accuracy : ",mean_squared_error(y_preds.reshape(-1,1),y_test.reshape(-1,1)))

    
    def predictor(self,value):
        if os.path.isfile(self.model_path) == True:
            #print("transform not called")
            loaded_model = pickle.load(open(self.model_path, 'rb'))
            y_pred = loaded_model.predict(np.array(value).reshape(-1,1))
            #print(y_pred[0][0])
            return y_pred[0][0] 

        else:
            #print("transform  called")
            self.transform()
            self.predictor(value)


#obj = Model()
#obj.transform()
#obj.predictor(121)