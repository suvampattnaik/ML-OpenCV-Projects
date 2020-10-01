  
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

data=pd.read_csv("datasets_180_408_data.csv")
data.drop(labels='Unnamed: 32',axis=1,inplace=True)
data['diagnosis'].replace('M',0,inplace=True)
data['diagnosis'].replace('B',1,inplace=True)
data.head(6)

x=data.iloc[:,2:33].values
y=data.iloc[:,1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

s=SimpleImputer(missing_values=0,strategy='mean')
x_train=s.fit_transform(x_train)
x_test=s.fit_transform(x_test)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

lr = DecisionTreeClassifier(random_state=0)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print("Accuracy is =",metrics.accuracy_score(y_test,y_pred))
