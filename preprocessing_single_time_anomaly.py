import pickle
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import plotly.graph_objects as go
import preprocessing_single_anomaly
import sys


def LSTM_MODEL():
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(X_train.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                        shuffle=False)

    X_train_pred = model.predict(X_train, verbose=0)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)


    threshold = np.max(train_mae_loss)
    print(f'Reconstruction error threshold: {threshold}')

    X_test_pred = model.predict(X_test, verbose=0)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)


    test_score_df = pd.DataFrame(test_2[TIME_STEPS:])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    test_score_df[columns[0]] = test_2[TIME_STEPS:][columns[0]]

    anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
    print(anomalies)

    previous_login = 0
    previous_date = ''

    max_val = test_score_df['y'].max()
    totalAnom = 0
    for index, row in test_score_df.iterrows():

        if row['anomaly'] == True and round(previous_login) == round(max_val):
            print(previous_date, "->1")
            totalAnom += 1
        elif row['anomaly'] == True and round(row['y']) == round(max_val):
            print(row['ds'], "->2")
            totalAnom += 1
        else:
            pass
        previous_login = row['y']
        previous_date = row['ds']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_score_df['ds'], y=test_score_df['loss'], name='Test loss'))
    fig.add_trace(go.Scatter(x=test_score_df['ds'], y=test_score_df['threshold'], name='Threshold'))
    fig.update_layout(showlegend=True, title='Test loss vs. Threshold')

    plot_data_json = fig.to_json()
    fig.show()

    return plot_data_json, totalAnom


'''
*************************************************************
BELOW THIS IS THE CODE FOR EVENTS PER DAY AND CATEGORY COUNT
*************************************************************
'''


def result_for_total_category(colName):
    category = []
    counts = []
    for cat in colName:
        if cat != 'Unnamed: 0' and cat != 'timestamp' and cat != 'message':
            Total = user_data[cat].sum()
            if Total != 0:
                category.append(cat)
                counts.append(int(Total))
    taskCategoryOverall = {'category': category, 'total': counts}
    return taskCategoryOverall


def result_for_events_per_day():

    # convert timestamps to dates and get count of unique dates
    dates = pd.to_datetime(user_data['timestamp'], unit='s').dt.date
    counts = dates.value_counts()

    # create dictionary with date values and their counts
    eventperday = {'date': pd.DatetimeIndex(counts.index).strftime('%Y-%m-%d').tolist(), 'total': counts.tolist()}

    return eventperday

#user = sys.argv[1]
user = 'we1775srv$'

user_data = preprocessing_single_anomaly.read_file(user)
cols = preprocessing_single_anomaly.get_columns(user_data)
dictionary = preprocessing_single_anomaly.separate(user_data, cols)
count_data = preprocessing_single_anomaly.hourly_frame_time(dictionary)
datetime_data = preprocessing_single_anomaly.convert_datetime(count_data)
mid_day, data_pass_split = preprocessing_single_anomaly.get_date_to_split(datetime_data)
train_1, test_1 = preprocessing_single_anomaly.splitting_data_train_test(mid_day, data_pass_split)
train_2, test_2 = preprocessing_single_anomaly.scaling(train_1, test_1)
columns = preprocessing_single_anomaly.get_col_new_csv(datetime_data)
X_train, y_train = preprocessing_single_anomaly.create_sequences(train_2[columns], train_2[columns[0]])
X_test, y_test = preprocessing_single_anomaly.create_sequences(test_2[columns], test_2[columns[0]])

TIME_STEPS = preprocessing_single_anomaly.TIME_STEPS

plotGraph, count = LSTM_MODEL()

taskCategoryOverall = result_for_total_category(cols)
print(taskCategoryOverall)
eventperday = result_for_events_per_day()
print(eventperday)

with open('plotGraphTime.pkl', 'wb') as f:
    pickle.dump([plotGraph, count, taskCategoryOverall, eventperday], f, protocol=3)
