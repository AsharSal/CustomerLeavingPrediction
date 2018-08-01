# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # : metrics of independ variables means take older lines and right of : is columns #take columns of index 3 to 12
y = dataset.iloc[:, 13].values #take last coloumn (dependent variable col) #metrics of dependent variable



# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])  # index 2 gender coloumn no need to hotencoder for this one
onehotencoder = OneHotEncoder(categorical_features = [1])  # France spain Germany indes 1 column
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense  # used for ANN step 1 which is randomly intialize the weights small no close to 0
# we will use sigmoid in output layer because it tells the probability & we want to calculate the probablity
# that specific person leaves bak or not.
# The best activation function is rectifier so we used it in Hidden layer.
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# we have 11 independent cols so 11 input nodes
classifier.add(Dense(output_dim = 6 , init = 'uniform', activation = 'relu',input_dim = 11))   # Every thing happens in dense func.

# Dense parameters 1) output_dim which is no of nodes in input layer
# tip: no of nodes in hidden layer can be choose by avg of input & output layers nodes. so 11+1/2 =6 nodes in hidden
# init used to intialize weights
# relu is basically rectifier

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6 , init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1 , init = 'uniform', activation = 'sigmoid'))
# if there are more categories in output layer you need to change output_dim and activation='softmax'

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# compile method takes optimizer which is gradient descent algorihm ,The mose effiecnt descent is adam.
#if there are more categories in output layer than loss = 'categorical_crossentropy'
# 3rd argument is metrics list having parameter accuracy to imporve accuracy on each epoch.

# Fitting the ANN to the Training set
# step 6 and step 7 of gradient descent.
# batch is no of observations after which you upadte the weights.
# find optimal batch size and epochs
classifier.fit(X_train,y_train, batch_size = 10, epochs = 50)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# to convert predicting probabilities into result of true or false we need to choose a threshold to decide when
# predicted result is 1 and when it is 0 natural threshod is 0.5 (50%)
y_pred =(y_pred > 0.5) # if larger than 0.5 return true 1 means person leaves the bank otherwise false 0
print (y_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print (cm)