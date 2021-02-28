import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class Spam:
    def __init__(self):
        self.model_path = "./static/assets/spam.sav"
        if os.path.isfile(self.model_path) == True:
            #print("not trained")
            pass
        else:   
            self.dataset = pd.read_csv('C:/Users/asus/.spyder-py3/spam.csv')

    def transform(self):
        corpus = []
        for i in range(0, 5572):
            review = re.sub('[^a-zA-Z]', ' ', self.dataset['EmailText'][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            corpus.append(review)
        cv = CountVectorizer(max_features = 10000)
        self.X = cv.fit_transform(corpus).toarray()
        self.y = self.dataset.iloc[:, 0].values
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.20, random_state = 0)
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        with open(self.model_path,"wb") as f:
            pickle.dump(classifier,f) 
            pickle.dump(cv,f)
         

    def predictor(self,msg):
        try:
            with open(self.model_path, "rb") as f:
                loaded_model = pickle.load(f)
                cv=pickle.load(f)
                res=loaded_model.predict(cv.transform([msg]).toarray())
                print(res)
                ans=0
                if res[0]=="spam":
                    ans=1
                else:
                    ans=0    
            print(ans)
            return int(ans)    
        except:
            self.transform()  
            self.predictor(msg)  
        