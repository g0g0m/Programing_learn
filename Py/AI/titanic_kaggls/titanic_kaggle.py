#Loading the support library
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

#Loading the traing,test data
train = pd.read_csv("/home/g0n/Documents/Programing_git/Py/AI/titanic_kaggls/train.csv")
test = pd.read_csv("/home/g0n/Documents/Programing_git/Py/AI/titanic_kaggls/test.csv")



#Befor Checking for missing data

print(train['Age'].isnull().values.sum() )
print(train['Age'].isnull().values.sum() / len(train['Age']) * 100)

print(test['Age'].isnull().values.sum() )
print(test['Age'].isnull().values.sum() / len(train['Age']) * 100)

#Null in mean
mean = np.mean(train['Age'])
train['Age'] = train['Age'].fillna(mean)

mean = np.mean(test['Age'])
test['Age']= test['Age'].fillna(mean)


#After Checking for missing data
print(train['Age'].isnull().values.sum() )
print(train['Age'].isnull().values.sum() / len(train['Age']) * 100)

print(test['Age'].isnull().values.sum() )
print(test['Age'].isnull().values.sum() / len(train['Age']) * 100)

#male=1, female=2
train['Sex'] = train['Sex'].str.replace('female', '2')
train['Sex'] = train['Sex'].str.replace('male', '1')

test['Sex'] = test['Sex'].str.replace('female', '2')
test['Sex'] = test['Sex'].str.replace('male', '1')

#ml_SVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score

x = pd.DataFrame({ 'Pclass':train['Pclass'], 'Sex':train['Sex'], 'Age':train['Age']})

y = pd.DataFrame({'Survived':train['Survived']})


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=None)

model = SVC(kernel='linear', random_state=None,C=0.1)
model.fit(x_train, y_train.values.ravel())


print(model.score(x_test,y_test))

xtest = pd.DataFrame({'Pclass':test['Pclass'], 'Sex':test['Sex'], 'Age':test['Age']})

print(model.predict(xtest))


