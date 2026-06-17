from flask import Flask , request , jsonify , render_template
from train_model import predict_price
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

CITIES = {
    'los_angeles':    {'name': 'Los Angeles',      'lat': 34.05,  'lon': -118.25},
    'san_francisco':  {'name': 'San Francisco',  'lat': 37.77,  'lon': -122.41},
    'san_diego':      {'name': 'San Diego',         'lat': 32.72,  'lon': -117.15},
    'san_jose':       {'name': 'San Jose',         'lat': 37.33,  'lon': -121.88},
    'sacramento':     {'name': 'Sacramento',       'lat': 38.57,  'lon': -121.47},
    'fresno':         {'name': 'Fresno',           'lat': 36.74,  'lon': -119.77},
}

@app.route('/')
def home():
    return render_template('index.html',cities=CITIES)

@app.route('/predict' , methods = ['POST'])
def predict():
    data = request.get_json()
    city_key = data['city']
    city = CITIES[city_key]

    result = predict_price(
        area_level = data['area_level'],
        house_age = float(data['house_age']),
        ave_rooms = float(data['ave_rooms']),
        ave_bedrms = float(data['ave_bedrms']),
        latitude = city['lat'],
        longitude = city['lon']
    )

    return jsonify({
        'mid':       result['mid'],
        'lower':     result['lower'],
        'upper':     result['upper'],
        'city_name': city['name']
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)