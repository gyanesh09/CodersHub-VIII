import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
import os 

class Sentiment:
    def __init__(self):
        self.model_path = "./static/assets/SentimentAnalyser.sav"
        if os.path.isfile(self.model_path) == True:
            #print("not trained")
            pass
        else:   
            dataset = pd.read_csv('C:/Users/asus/.spyder-py3/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    def transform(self):
        corpus = []
        dataset = pd.read_csv('C:/Users/asus/.spyder-py3/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
        for i in range(0, 1000):
            review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
            review = review.lower()
            review = review .split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            corpus.append(review)

        # Creating the Bag of Words model
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, 1].values    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        # Fitting Naive Bayes to the Training set
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        print(X_train.shape)
        with open(self.model_path,"wb") as f:
            pickle.dump(classifier,f) 
            pickle.dump(cv,f) 

    def predictor(self,value):
        if os.path.isfile(self.model_path) == True:
            corpa=[]
            query = re.sub('[^a-zA-Z]', ' ', value)
            query = query.lower()
            query = query.split()
            ps = PorterStemmer()
            query = [ps.stem(word) for word in query if not word in set(stopwords.words('english'))]
            query = ' '.join(query)
            corpa.append(query)
            with open(self.model_path, "rb") as f:
                testout1 = pickle.load(f)
                testout2 = pickle.load(f)
                #print(testout1,testout2)
            custstringtest= testout2.transform(corpa).toarray()
            y_pred = testout1.predict(custstringtest)
            return int(y_pred)
        else:
            self.transform()
            self.predictor()