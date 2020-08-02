import flask
import pickle
with open(f'model\prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
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
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html', original_input = {'Hurricane Latitude':Hlat, 'Hurricane Longitude':Hlong, 'Maximum Sustained Winds (in knots)':MaxSusWinds, 'Country Latitude':Clat, 'Country Longitude':Clong, 'Country Population':Pop, 'Country Area (in square km)':Area}, result = prediction,)

if __name__ == '__main__':
    app.run()
