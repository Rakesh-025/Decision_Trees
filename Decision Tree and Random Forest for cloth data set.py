#Decision tree and random forest for cloth manufacturing data set
#Importing the packages

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame

#Importing the data set into python
cloth_data = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\Assignments\Assignment Datasets\Decision Tree\Company_Data.csv")

#Finding any missing values in the data set
cloth_data.isnull().sum()
cloth_data.dropna()

#Segregating the sales as yes and no if its value is greater that 9 yes or else no
highsales = DataFrame(cloth_data.Sales)
highsales.loc[highsales['Sales']<9,'highsales'] = 'No'
highsales.loc[highsales['Sales']>=9,'highsales'] = 'Yes'
highsales

#Creating dummy variable for the catagorical data
lb = LabelEncoder()
cloth_data["ShelveLoc"] = lb.fit_transform(cloth_data["ShelveLoc"])
cloth_data["Urban"] = lb.fit_transform(cloth_data["Urban"])
cloth_data["US"] = lb.fit_transform(cloth_data["US"])

highsales['highsales'].unique()
highsales['highsales'].value_counts()

#Segregating the data set as output and input data sets
predictors = cloth_data.drop(["Sales"], axis = 1)
target = highsales.drop(["Sales"], axis = 1)

#Splitting the data into taining and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.3, random_state=0)

#Applying decision tree for the data set
from sklearn.tree import DecisionTreeClassifier as DT

#Creating the decision tree model for the data set
model = DT(criterion = 'entropy')
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Finding  the accuracy of the model with the testing data
confusion_matrix(y_test, model.predict(x_test))
accuracy_score(y_test, model.predict(x_test))

# Finding  the accuracy of the model with the training data
confusion_matrix(y_train, model.predict(x_train))
accuracy_score(y_train, model.predict(x_train))

#Applying random forest model to the data set
from sklearn.ensemble import RandomForestClassifier

#creating the radom forest model on the data set
random_forest_clsf = RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=42)

random_forest_clsf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Finding  the accuracy of the random forest model with the testing data
confusion_matrix(y_test, random_forest_clsf.predict(x_test))
accuracy_score(y_test, random_forest_clsf.predict(x_test))

# Finding  the accuracy of the random forest model with the training data
confusion_matrix(y_train, random_forest_clsf.predict(x_train))
accuracy_score(y_train, random_forest_clsf.predict(x_train))
