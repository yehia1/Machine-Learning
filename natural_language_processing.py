# -*- coding: utf-8 -*-
"""Natural language processing

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b8XV-DDSEarAN4-b5DBw21lkUpzXaGA-
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv('Restaurant_Reviews.tsv' ,delimiter='\t',quoting = 3)
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
reward = []
for i in range(1000):
  reward.append(re.sub('[^a-zA-Z]',' ',file['Review'][i]))
  reward[i] = reward[i].lower()
  reward[i] = reward[i].split()
  ps = PorterStemmer()
  reward[i] = [ps.stem(word) for word in reward[i] if not word in set(stopwords.words('english'))]
  reward[i] = ' '.join(reward[i])
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(reward).toarray()
y = file.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

#making the random forest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion = 'entropy' ,random_state = 100)
classifier.fit(x_train,y_train)
y_predict = classifier.predict(x_test)

'''from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(RandomForestClassifier(), {
    'n_estimators': [10,50,100,250],
    'criterion': ['gini','entropy'],
    'random_state': [100,500,1000]
    }, cv=5, return_train_score=False)
clf.fit(x_train,y_train)
print(clf.best_score_)
print(clf.best_params_)'''


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))