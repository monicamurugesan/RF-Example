# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:05:23 2020

@author: Hp
"""

# Random Forest Classifier

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets

datasets = pd.read_csv('H:\\RStudio\\assignment\\assignments1\\Random Forest\\Fraud_check.csv')
datasets["Taxable.Income"]
datasets.loc[datasets['Taxable.Income']<=30000,'Taxable']="RISKY"
datasets.loc[datasets['Taxable.Income']>30000,'Taxable']="GOOD"

datasets["Taxable"]=datasets["Taxable"].replace({"GOOD":0,"RISKY":1})
datasets["Urban"]=datasets["Urban"].replace({"YES":0,"NO":1})
datasets["Undergrad"]=datasets["Undergrad"].replace({"YES":0,"NO":1})
datasets["Marital.Status"]=datasets["Marital.Status"].replace({"Single":0,"Married":1,"Divorced":2})
datasets=datasets.drop(['Taxable.Income'],axis=1)
datasets.head()
X = datasets.iloc[:, [0,1,2,3,4]].values
Y = datasets.iloc[:,5].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.5, random_state =1000)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_Train,Y_Train)

# Predicting the test set results

Y_Pred = classifier.predict(X_Test)

# Making the Confusion Matrix 

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm = confusion_matrix(Y_Test, Y_Pred)
acc1=accuracy_score(Y_Test,Y_Pred)
print(acc1)
class1=classification_report(Y_Test,Y_Pred)
print(class1)
# Visualising the Training set results

# Visualising the Test set results
len(X_Train)
len(X_Test)
len(Y_Train)
len(Y_Test)
classifier.estimators_

featureimp=pd.Series(classifier.feature_importances_).sort_values(ascending=True)
print(featureimp)
sns.barplot(x=round(featureimp,4),y=featureimp)
plt.xlabel("Feature importance")
plt.show()
