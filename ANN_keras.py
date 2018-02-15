#Python 3.6.0
# Importing the libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import os


# 1.Load Data and Clean Data

# Path to working directory

os.chdir("Path to working directory")

# Importing the dataset
data = pd.read_csv('banking.csv', header=0)
data = data.dropna()
#print(data.shape)
#print(list(data.columns))

# We will drop the variables that we do not need.
data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)

# Create dummy variables
data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])

#print(data2.shape)

#2. Data Preprocessing

X = data2.iloc[:, 1:29].values
y = data2.iloc[:, 0].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# creat ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
# Adding the first hidden layer
classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 28))

# Adding the second hidden layer

classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

##acc: 0.8972



# Lets test on test dataset
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

##([[7242,   83],[ 740,  173]]


# checking accracy using K-fold cross validation

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 28))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
## mean = 0.89632776225825184
variance = accuracies.std()
## variance = 0.0026693334367559221


