import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

path = "11 groups_whole range.xlsx"
total = pd.read_excel(path)
total = total.T
total = total.values
X = total[:, 1:].astype(float)

y = total[:, 0]
enc = LabelEncoder()
y = enc.fit_transform(y)

permutation = list(np.random.permutation(X.shape[0]))
X = X[permutation, :]
y = y[permutation]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

mysvm = svm.SVC(C=1, kernel='linear')
mysvm.fit(X_train, y_train)
result_svm = mysvm.predict(X_test)


print(accuracy_score(y_test, result_svm))