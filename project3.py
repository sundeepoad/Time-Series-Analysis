
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# importing csv
df = pd.read_csv("tbl_ChartPerio.csv")

# Removing Null values  in specific columns
data = df.dropna(subset=["fld_strClassification", "fld_strHygiene", "fld_strCalculus", "fld_strExamType"])

# Specifying labels and features
my_label = data["fld_strClassification"]  ## label column for  classification algorithm
my_features = data[["fld_strHygiene", "fld_strCalculus", "fld_strExamType"]]  ## features


## To check what each label is getting encoded to.
print("Before encoding", my_label[:5])

# Encoding labels
l= LabelEncoder()
my_label = l.fit_transform(my_label)

## To check what each label is getting encoded to.
print("After encoding", my_label[:5])

### encoding features
for i in my_features.columns:
    my_features[i] = LabelEncoder().fit_transform(my_features[i])


# Dividing data into training and testing. 20% for testing, 80% for training

X_t, X_test, y_t, y_test = train_test_split(my_features, my_label, test_size = 0.2, random_state = 34)



### Defining decison tree clasffication algorithm
decision_tree = DecisionTreeClassifier()

## Fitting classifier
decision_tree.fit(X_t, y_t)

# Making predictions 
pred = decision_tree.predict(X_test)

# Evaluating performance of Model
acc = accuracy_score(y_test, pred)
print("Accuracy of Decision Tree:", acc)


##############

### APPLYING KNN CLASSIFIER

## Dividing data into testing and training. 85% for training and 15% for testing. Model didn't perform good with 80% training data
X_tr, X_te, y_tr, y_te = train_test_split(my_features, my_label, test_size=0.15, random_state=28)



# Defining Classifier with K =5
## I tried different k values and checked accuracy. K <= 5 gives suitable accuracy. So I chose  K=5
knn = KNeighborsClassifier(n_neighbors=5)

# Fitting the model
knn.fit(X_tr, y_tr)

# Making predictions
predic = knn.predict(X_te)

# Finding accuracy of model
accuracy = accuracy_score(y_te, predic)
print("Accuracy of KNN: ", accuracy)


### Making prediction on my own chosen value

lst = [3,3,1]
ipt = np.array(lst).reshape(1,-1)
res = knn.predict(ipt)   ## predicting on KNN model
res2 = decision_tree.predict(ipt)  ## predictin on Decision Tree model

print("my prediction", res, res2) ## Both give same results
