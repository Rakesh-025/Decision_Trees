#Decision tree and random forest for Fraud Check data set
#Importing the packages

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame

#Importing the data set into python
check_data = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\Assignments\Assignment Datasets\Decision Tree\Fraud_check.csv")

#Finding any missing values in the data set
check_data.isnull().sum()
check_data.dropna()
check_data.columns

#Segregating the taxable income as good and risky
risky_good = DataFrame(check_data["Taxable.Income"])
risky_good.loc[risky_good['Taxable.Income']<=30000,'risky_good'] = 'Risky'
risky_good.loc[risky_good['Taxable.Income']>30000,'risky_good'] = 'Good'
risky_good

#Creating dummy variable for the catagorical data
lb = LabelEncoder()
check_data["Undergrad"] = lb.fit_transform(check_data["Undergrad"])
check_data["Marital.Status"] = lb.fit_transform(check_data["Marital.Status"])
check_data["Urban"] = lb.fit_transform(check_data["Urban"])

risky_good['risky_good'].unique()
risky_good['risky_good'].value_counts()

#Segregating the data set as output and input data sets
predictors = check_data.drop(["Taxable.Income"], axis = 1)
target = risky_good.drop(["Taxable.Income"], axis = 1)

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
random_forest_clsf = RandomForestClassifier(n_estimators=250, n_jobs=1, random_state=42)

random_forest_clsf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Finding  the accuracy of the random forest model with the testing data
confusion_matrix(y_test, random_forest_clsf.predict(x_test))
accuracy_score(y_test, random_forest_clsf.predict(x_test))

# Finding  the accuracy of the random forest model with the training data
confusion_matrix(y_train, random_forest_clsf.predict(x_train))
accuracy_score(y_train, random_forest_clsf.predict(x_train))
