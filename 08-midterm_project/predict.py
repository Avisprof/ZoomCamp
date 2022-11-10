import pickle

from flask import Flask
from flask import request
from flask import jsonify

import xgboost as xgb

model_file = 'xgb_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('SupermarketSales')
@app.route('/predict', methods=['POST'])
def predict():

    lag_sales = request.get_json()
    print(lag_sales)

    X = dv.transform([lag_sales])
    print(X)
    dmatrix = xgb.DMatrix(X)

    y_pred = model.predict(dmatrix).round(0)


    result = {'product_line': lag_sales['product_line'],
              'city': lag_sales['city'],
              'predict_sales': int(y_pred)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)