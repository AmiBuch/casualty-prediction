import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import xgboost as xgb
import flask
#reading my csv file from the url
data = pd.read_csv("https://raw.githubusercontent.com/AmiBuch/HurricaneKat-python-ml-data/master/HurricaneDat.csv")
#specifying the data on which the casualties/deaths are predicted
x = data[['Hlat', 'Hlong', 'MaxSusWinds', 'Clat', 'Clong', 'Pop', 'Area']]
y = data['Death']
#training my model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 1)
clf = xgb.sklearn.XGBClassifier(nthread = -1, seed = 1)
clf.fit(x_train, y_train)

app = flask.Flask(__name__, template_folder = 'templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request_method == 'POST':
        Hlat = flask.request.form['Hlat']
        Hlong = flask.request.form['Hlong']
        MaxSusWinds = flask.request.form['MaxSusWinds']
        Clat = flask.request.form['Clat']
        Clong = flask.request.form['Clong']
        Pop = flask.request.form['Pop']
        Area = flask.request.form['Area']
        input_variables = pd.DataFrame([['Hlat', 'Hlong', 'MaxSusWinds', 'Clat', 'Clong', 'Pop', 'Area']], columns = [['Hlat', 'Hlong', 'MaxSusWinds', 'Clat', 'Clong', 'Pop', 'Area']], dtype=float)
        prediction = clf.predict(input_variables)
        return flask.render_template('main.html', original_input = {'Hurricane Latitude':Hlat, 'Hurricane Longitude':Hlong, 'Maximum Sustained Winds (in knots)':MaxSusWinds, 'Country Latitude':Clat, 'Country Longitude':Clong, 'Country Population':Pop, 'Country Area (in square km)':Area}, result = prediction,)

if __name__ == '__main__':
    app.run()
#model_pickle = "C:\Desktop\webapp\model\prediction_model.pkl"
#with open(model_pickle, 'wb') as file:
 #   pickle.dump(clf, file)




