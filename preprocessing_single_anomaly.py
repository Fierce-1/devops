import pandas as pd
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


def read_file(user):
    df = pd.read_csv(user + '.csv')
    return df


def get_columns(df):
    columns = {}
    for index, col in enumerate(df.columns):
        columns[col] = index
    return columns


def separate(df, col):

    diction = {'time': [], 'logon': []}

    for index, row in df.iterrows():
        diction['time'].append(float(row[col['timestamp']]))
        diction['logon'].append(row[col['Logon']])

    return diction


def hourly_frame_count(df):
    times = df['timestamp']
    start_time = times[0]
    end_time = start_time + (1 * 60 * 60)
    event_log = {}

    for time in times:
        current_time = time
        if current_time < end_time and start_time in event_log:
            event_log[start_time] = event_log[start_time] + 1
        elif current_time < end_time:
            event_log[start_time] = 1
        else:
            start_time = current_time
            end_time = start_time + (1 * 60 * 60)
            event_log[start_time] = 1

    time = []
    counts = []
    for key, value in event_log.items():
        time.append(datetime.utcfromtimestamp(key))
        counts.append(value)

    dict = {'ds': time, 'y': counts}

    df1 = pd.DataFrame(dict)

    # saving the dataframe
    print(df1)
    return df1


def hourly_frame_time(diction):

    times = diction['time']
    logon = diction['logon']
    # print(dict_1['time'][0],type(dict_1['time'][0]))
    start_time = float(diction['time'][0])
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
    dict = {'ds': time, 'y': counts}

    df1 = pd.DataFrame(dict)
    print(df1)
    return df1


def convert_datetime(df1):

    df1['ds'] = pd.to_datetime(df1['ds'])

    return df1


def get_date_to_split(df1):

    start_date_count = df1['ds'].min()
    string_start_year_count = start_date_count.strftime('%Y')
    string_start_day_count = start_date_count.strftime('%d')
    string_start_month_count = start_date_count.strftime('%m')
    int_start_day_count = int(string_start_day_count)

    end_date_count = df1['ds'].max()
    string_end_year_count = end_date_count.strftime('%Y')
    string_end_day_count = end_date_count.strftime('%d')
    string_end_month_count = end_date_count.strftime('%m')
    int_end_day_count = int(string_end_day_count)

    mid_day_count = (int_start_day_count + int_end_day_count) // 2

    mid_day_count = str(mid_day_count)

    mid_day_count = string_start_year_count + '-' + string_start_month_count + '-' + mid_day_count

    return mid_day_count, df1


def get_col_new_csv(df2):

    columns = []

    for index, col in enumerate(df2.columns):
        if index >= 1:
            columns.append(col)

    return columns


def splitting_data_train_test(day, df3):
    # count
    train, test = df3.loc[df3['ds'] > day], df3.loc[df3['ds'] < day]

    return train, test


def scaling(train, test):

    # count
    scaler = StandardScaler()
    scaler = scaler.fit(train[['y']])

    train['y'] = scaler.transform(train[['y']])
    test['y'] = scaler.transform(test[['y']])

    return train, test


TIME_STEPS = 1


def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []

    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)


