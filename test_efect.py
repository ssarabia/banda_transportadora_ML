from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

dataX  = pd.read_csv('./dataset/train_x.csv')
y_actu = pd.read_csv('./dataset/train_y.csv')

clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(dataX, np.ravel(y_actu))
data_test_X = pd.read_csv('./dataset/test_x.csv')
y_pred = clf.predict(data_test_X)

cnf = confusion_matrix(y_actu, y_pred)


print(cnf)
