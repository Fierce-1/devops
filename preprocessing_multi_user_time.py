from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import tensorflow as tf
import numpy as np
from datetime import datetime

np.random.seed(1)
tf.random.set_seed(1)
print('Tensorflow version: ', tf.__version__)


def read_file(user):
    data_frame = pd.DataFrame(dictionary_user_1[user])
    return data_frame


def separate(df, diction_time, user, diction_time_user):
    columns = {}
    for index, col in enumerate(df.columns):
        if index >= 1:
            columns[col] = index

    for index, row in df.iterrows():
        diction_time['time'].append(float(row[columns['timestamp']]))
        diction_time[user].append(row[columns['Logon']])

    for index, i in enumerate(diction_time):
        if i not in diction_time_user:
            diction_time_user[i] = diction_time[i]

    return diction_time_user


def hourly_frame_time(dict_user, user, dictionary_time):

    times = dict_user['time']
    logon = dict_user[user]
    # print(dict_1['time'][0],type(dict_1['time'][0]))
    start_time = float(dict_user['time'][0])
    end_time = start_time + (1 * 60 * 60)
    event_log = {}

    for index, time in enumerate(times):
        current_time = time

        if current_time < end_time and start_time in event_log:
            if logon[index] == 1:
                event_log[start_time] = 1
        elif current_time < end_time:
            if logon[index] == 1:
                event_log[start_time] = 1

        else:
            start_time = current_time
            end_time = start_time + (1 * 60 * 60)
            if logon[index] == 1:
                event_log[start_time] = 1
            else:
                event_log[start_time] = 0

    time = []
    counts = []
    for key, value in event_log.items():
        time.append(datetime.utcfromtimestamp(key))
        counts.append(value)

    # importing the module
    # print(event_log)
    dict = {'ds': time, user: counts}

    for index, i in enumerate(dict):
        if i not in dictionary_time:
            dictionary_time[i] = dict[i]

    return dictionary_time


def convert_data_frame(dictionary):
    dataframe = pd.DataFrame(dictionary)
    print(dataframe)
    return dataframe


def get_data_split(df):
    df['ds'] = pd.to_datetime(df['ds'])

    start_date = df['ds'].min()
    string_start_year = start_date.strftime('%Y')
    string_start_day = start_date.strftime('%d')
    string_start_month = start_date.strftime('%m')
    int_start_day = int(string_start_day)

    end_date = df['ds'].max()
    string_end_year = end_date.strftime('%Y')
    string_end_day = end_date.strftime('%d')
    string_end_month = end_date.strftime('%m')

    int_end_day = int(string_end_day)

    mid_day = (int_start_day + int_end_day) // 2

    mid_day = str(mid_day)

    mid_day = string_start_year + '-' + string_start_month + '-' + mid_day

    return mid_day


def split_train_test(mid_day, df):
    train, test = df.loc[df['ds'] >= mid_day], df.loc[df['ds'] < mid_day]

    return train, test


def scaling(train, test, df):

    columns = []

    for index, col in enumerate(df.columns):
        if index >= 1:
            columns.append(col)

    df_training = train[columns]
    df_testing = test[columns]

    scaler = StandardScaler()
    scaler = scaler.fit(df_training)

    df_training_scaled = scaler.transform(df_training)
    df_testing_scaled = scaler.transform(df_testing)

    scaled_df_training = pd.DataFrame(df_training_scaled, columns=columns)
    scaled_df_testing = pd.DataFrame(df_testing_scaled, columns=columns)
    scaled_df_training['ds'] = train['ds'].values
    scaled_df_testing['ds'] = test['ds'].values

    scaled_df_training = scaled_df_training.iloc[:, [2, 0, 1]]
    scaled_df_testing = scaled_df_testing.iloc[:, [2, 0, 1]]

    return scaled_df_training, scaled_df_testing


TIME_STEPS=1


def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []

    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])

    return np.array(Xs), np.array(ys)


def model_LSTM(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(X_train.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    history = model.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()

    X_train_pred = model.predict(X_train, verbose=0)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

    plt.hist(train_mae_loss, bins=50)
    plt.xlabel('Train MAE loss')
    plt.ylabel('Number of Samples')

    threshold = np.max(train_mae_loss)
    print(f'Reconstruction error threshold: {threshold}')

    X_test_pred = model.predict(X_test, verbose=0)
    test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

    plt.hist(test_mae_loss, bins=50)
    plt.xlabel('Test MAE loss')
    plt.ylabel('Number of samples')

    test_score_df = pd.DataFrame(scaled_df_testing[TIME_STEPS:])

    loss = {}

    for i in columns:
        loss['loss_' + i] = []

    users = []

    for j in loss:
        users.append(j)


    for k in test_mae_loss:
        for index, l in enumerate(k):
            user = users[index]
            loss[user].append(l)

    for loss_user in users:
        test_score_df[loss_user] = loss[loss_user]


    columns_test_score = {}
    for index, col_test_score in enumerate(test_score_df.columns):
        columns_test_score[col_test_score] = index + 1


    anom = []
    for row in test_score_df.itertuples():
        is_anomaly = False
        for m in range(columns_test_score[users[0]], len(columns_test_score) + 1):
            if row[m] > threshold:
                is_anomaly = True
                break

        anom.append(is_anomaly)
    test_score_df['anomaly'] = anom
    print(test_score_df)
    anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
    print(anomalies)

    previous_login=0
    previous_date=''
    #print(test_score_df)


    for i in columns:
        max_val = test_score_df[i].max()
        print('\nUser ' + i + ' Anomalies')



        for index, row in test_score_df.iterrows():
            #     print(index)
            if row['anomaly'] == True and round(previous_login) == round(max_val):
                #         print(test_score_df[index+1:index+672])
                print(previous_date,"->1")
            elif row['anomaly'] == True and round(row[i]) == round(max_val):
                print(row['ds'],"->2")
            else:
                pass
            previous_login=row[i]

            previous_date=row['ds']


if __name__ == '__main__':
    file = "users/we1775srv$.csv"
    df1 = pd.read_csv(file)
    df1.drop('Unnamed: 0', axis=1, inplace=True)
    dictionary_user = df1.to_dict(orient='list')
    dictionary_time_user = {}
    dictionary_time = {}

    dictionary_user_1 = {'we1775srv$': dictionary_user, 'we1550srv$': dictionary_user}

    for user in dictionary_user_1:
        diction = {'time': [], user: []}
        data_frame = read_file(user)
        diction_user = separate(data_frame, diction, user, dictionary_time_user)
        hourly_frame_time(diction_user, user, dictionary_time)

    dataframe = convert_data_frame(dictionary_time)
    mid_day = get_data_split(dataframe)
    train, test = split_train_test(mid_day, dataframe)
    scaled_df_training, scaled_df_testing = scaling(train, test, dataframe)

    columns = []

    for index, col in enumerate(dataframe.columns):
        if index >= 1:
            columns.append(col)

    X_train, y_train = create_sequences(scaled_df_training[columns], scaled_df_training[columns])
    X_test, y_test = create_sequences(scaled_df_testing[columns], scaled_df_testing[columns])

    model_LSTM(X_train, y_train, X_test, y_test)

