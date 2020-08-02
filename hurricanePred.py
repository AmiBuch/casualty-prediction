import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import ensemble
import pickle

#reading my csv file from the url
data = pd.read_csv("https://raw.githubusercontent.com/AmiBuch/HurricaneKat-python-ml-data/master/HurricaneDat.csv")
#specifying the data on which the casualties/deaths are predicted
x = data[['Hlat', 'Hlong', 'MaxSusWinds', 'Clat', 'Clong', 'Pop', 'Area']]
y = data['Death']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 1)
reg = linear_model.LinearRegression()
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')
clf.fit(x_train, y_train)
model_pickle = "C:\Desktop\webapp\model\prediction_model.pkl"
with open(model_pickle, 'wb') as file:
    pickle.dump(clf, file)




