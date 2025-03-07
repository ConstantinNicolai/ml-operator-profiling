# importing required libraries
# importing Scikit-learn library and datasets package
from sklearn import datasets

# # Loading the iris plants dataset (classification)
# iris = datasets.load_iris()

# print(iris.target_names)

# print(iris.feature_names)

# dividing the datasets into two parts i.e. training datasets and test datasets
X, y = datasets.load_iris( return_X_y = True)


# Splitting arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# importing random forest classifier from assemble module
from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
# # creating dataframe of IRIS dataset
# data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1],
# 					'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],
# 					'species': iris.target})


# # printing the top 5 datasets in iris dataset
# print(data.head())


# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)  
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
 
# metrics are used to find accuracy or error
from sklearn import metrics  
print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))



# predicting which type of flower it is.
print(clf.predict([[3, 3, 5, 2]]))
