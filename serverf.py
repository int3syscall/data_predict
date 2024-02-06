from flask import Flask, jsonify
from test1_working_model_copy import predict
from fetch_data import fetch_datas , fetch_datas_last_24
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/data', methods=['GET'])
def get_prediction():
    try:
        last_day_data = fetch_datas("80.66.87.47", 3306, "laveraluser", "testdata2s", "root", "Password123!jj")
        prediction = predict(last_day_data)
        return jsonify(prediction)
    except Exception as e:
        return [{"error": "something went wrong!"}]

@app.route('/data24', methods=['GET'])
def get_prediction_today():
    try:
        last_day_data = fetch_datas_last_24("80.66.87.47", 3306, "laveraluser", "testdata2s", "root", "Password123!jj")
        if len(last_day_data) < 1:
            return [{"error": "no data available for the last 24 hours!"}]
        prediction = predict(last_day_data)
        return jsonify(prediction)
    except Exception as e:
        return [{"error": "something went wrong!"}]
    
@app.route('/', methods=['GET'])
def home():
    return "Actually there is no homepage for this API, please use /data or /data24 to get the data."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8055)