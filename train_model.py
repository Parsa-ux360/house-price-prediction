from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , r2_score
import numpy as np
import pandas as pd
import joblib

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns= housing.feature_names)
df['Price'] = housing.target

features = ['MedInc', 'HouseAge' , 'AveRooms' , 'AveBedrms', 'Latitude' , 'Longitude']

X = df[features]
y = df['Price']

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= 0.2 , random_state= 42)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train , y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

joblib.dump(model , 'model.pkl')
print('Model saved successfully ✅')

AREA_LEVELS = {
    'laxury' : 12.0,
    'medium' : 7.0,
    'cheap' : 2.5,
}

RMSE_VALUE = rmse

def predict_price(area_level,house_age , ave_rooms , ave_bedrms , latitude , longitude):
    med_inc = AREA_LEVELS[area_level]
    input_data = pd.DataFrame([[med_inc,house_age , ave_rooms , ave_bedrms, latitude , longitude]] , columns=features)
    predicted = model.predict(input_data)[0]

    mid   = round(predicted * 100_000, 2)
    lower = round((predicted - RMSE_VALUE) * 100_000, 2)
    upper = round((predicted + RMSE_VALUE) * 100_000, 2)

    lower = max(lower, 0)
    mid   = max(mid, 0)

    return {'mid': mid , 'lower': lower, 'upper': upper}