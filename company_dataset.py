# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

c1=pd.read_csv("H://RStudio//assignment//assignments1//Random Forest//Company_Data.csv")
c1.head()
c1.loc[c1['Sales']<10,'high sales']=0
c1.loc[c1['Sales']>10,'high sales']=1
c1["Urban"]=c1["Urban"].replace({"Yes":0,"No":1})
c1["US"]=c1["US"].replace({"Yes":1,"No":0})
c1["ShelveLoc"]=c1["ShelveLoc"].replace({"Bad":0,"Medium":1,"Good":2})
X=c1.iloc[:,[1,2,3,4,5,6,7,8,9]]
Y=c1.iloc[:,10]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,Y_train)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
pred=classifier.predict(X_test)
accuracy=accuracy_score(Y_test,pred)
print("Accuracy:",accuracy*100)
cm=confusion_matrix(Y_test,pred)
cm
cls1=classification_report(Y_test,pred)
print(cls1)
sns.pairplot(c1)
classifier.estimators_
sns.heatmap(cm, annot=True)
import graphviz 
dot_data = tree.export_graphviz(clf, out_file='tree.dot')
classifier.feature_importances_
featureimp=pd.Series(classifier.feature_importances_).sort_values(ascending=True)
print(featureimp)
sns.barplot(x=round(featureimp,4),y=featureimp)
plt.xlabel("Feature importance")
plt.show()
i