#Decision tree and random forest for Diabeties data set
#Importing the packages

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


#Importing the data set into python
diabeties_data = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\Assignments\Assignment Datasets\Decision Tree\Diabetes.csv")

#Finding any missing values in the data set
diabeties_data.isnull().sum()
diabeties_data.dropna()

lb = LabelEncoder()
diabeties_data[" Class variable"] = lb.fit_transform(diabeties_data[" Class variable"])
diabeties_data[' Class variable'].unique()
diabeties_data[' Class variable'].value_counts()
colnames = list(diabeties_data.columns)
predictors = colnames[:8]
target = colnames[8]


# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(diabeties_data, test_size = 0.3)

#Applying decision tree for the data set
from sklearn.tree import DecisionTreeClassifier as DT

#Creating the decision tree model for the data set
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction and Finding  the accuracy of the model with the testing data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target])

# Prediction and Finding  the accuracy of the model with the training data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) 


#Applying random forest model to the data set
from sklearn.ensemble import RandomForestClassifier

#creating the radom forest model on the data set
random_forest_clsf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

random_forest_clsf.fit(train[predictors], train[target])

# Prediction and Finding  the accuracy of the random forest model with the testing data
preds = random_forest_clsf.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target])

# Prediction and Finding  the accuracy of the random forest model with the training data
preds = random_forest_clsf.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target])
