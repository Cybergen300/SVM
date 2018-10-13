"""
Machine learning part 1 : support vector machine tuto
"""

import os 
path= "\your path"
os.chdir(path)
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
                   header = None)

data.columns = ["variance", "skewness", "curtosis", "entropy","Class"]

X = data.drop('Class', axis= 1)
y = data['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.svm import SVC
svclassifier = SVC(kernel= 'linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
