# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:41:02 2020

@author: Hp
"""
####Loading Libraries..
import pandas as pd
import numpy as np1
import seaborn as sns
import matplotlib.pyplot as plt

#### Loading Datasets
from sklearn.datasets import load_iris

iris=load_iris()


dfX=pd.DataFrame(iris.data,columns=iris.feature_names)
dfX
dfY=pd.DataFrame(iris.target)
dfY
df=pd.DataFrame([dfX,dfY])
###Train and Test...
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dfX,dfY, test_size=0.5, stratify=iris.target, random_state=123456)

###Creating model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_jobs=4,n_estimators=100,criterion="entropy",oob_score=True)
rf.fit(X_train,y_train)
rf.predict(X_train)

###Predicting and calculate accuracy,confusion matrix ,classification report
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
predict=rf.predict(X_test)
accuracy=accuracy_score(y_test,predict)
print("Accuracy:",accuracy*100)
print("OOB Score:",rf.oob_score_)
cm=confusion_matrix(y_test,predict)

sns.heatmap(cm)
pr=classification_report(y_test,predict)

sns.pairplot(df)
