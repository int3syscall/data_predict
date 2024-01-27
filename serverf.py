from flask import Flask, jsonify
from test1_working_model_copy import predict
from fetch_data import fetch_datas

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def get_prediction():
    last_day_data = fetch_datas("localhost", 3306, "laveraluser", "testdata2s")
    prediction = predict(last_day_data)
    prediction = str(prediction).replace("},{", "},\n{")
    # return jsonify(prediction)
    return prediction

@app.route('/test_api', methods=['GET'])
def test_api():
    data = {"test": "Hello World api test"}
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8055)