import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
# from fetch_data import fetch_datas
import os
import json



def predict(data):

    # Load the data
    df = data #pd.read_csv(data)

    # Convert 'Date' and 'Time' columns to datetime type
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')

    # Use 'watt' column for prediction
    data = df['watt'].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Split the data into training and testing data
    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size], data[train_size:]

    # Convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    # Reshape into X=t and Y=t+1
    look_back = 1
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Check if the model is already trained
    if os.path.isfile('model.h5'):
        # Load the trained model
        model = load_model('model.h5')
    else:
        # Build and train the model
        model = Sequential()
        model.add(LSTM(4, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, Y_train, epochs=10, batch_size=1, verbose=2)

        # Save the trained model
        model.save('model.h5')

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invert predictions back to original scale
    train_predict = scaler.inverse_transform(train_predict)
    Y_train = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = scaler.inverse_transform([Y_test])

    # Plot the results
    # plt.figure(figsize=(8,4))
    # plt.plot(np.append(Y_train, Y_test), label='Observed', color='#006699');
    # plt.plot(np.append(train_predict, test_predict), label='Prediction', color='#ff0066');
    # plt.legend(loc='upper left')
    # plt.title('LSTM Recurrent Neural Net')
    # plt.show()

    # Create a DataFrame from the predicted values
    df_predict = pd.DataFrame(np.append(train_predict, test_predict), columns=['Predicted'])

    # Get the last datetime in the original data
    last_datetime = df.index[-1]

    # Generate a date range for the next day, each second
    next_day = pd.date_range(start=last_datetime + pd.Timedelta(seconds=1), periods=len(df_predict), freq='S')

    # Convert the DatetimeIndex to a string representation
    next_day_str = next_day.strftime('%Y-%m-%d %H:%M:%S')

    # Create a DataFrame from the predicted values
    df_predict = pd.DataFrame({
        'Datetime': next_day_str,
        'Predicted': np.append(train_predict, test_predict)
    })

    # Convert the 'Datetime' column to a string
    # df_predict['Datetime'] = df_predict['Datetime'].astype(str)


    # # Create a DataFrame from the predicted values
    # df_predict = pd.DataFrame({
    #     'Datetime': next_day,
    #     'Predicted': np.append(train_predict, test_predict)
    # })

    # Save the DataFrame to a CSV file
    # df_predict.to_csv('predicted_data.csv', index=False)

    json_data = df_predict.to_json(orient='records')

    json_data.replace('[', '')
    json_data.replace(']', '')

    data_as_dic = json.loads(json_data)

    return data_as_dic


# last_day_data = fetch_datas("80.66.87.47", 3306, "laveraluser", "testdata2s", "root", "Password123!jj")
# predict(last_day_data)