import pickle
import sys
from datetime import datetime
from prophet import Prophet

# Data processing
import pandas as pd

# Model performance evaluation
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def read_file(user):
    data_frame = pd.DataFrame(dictionary_user_1[user])
    return data_frame


def hourly_frame_count_multi(dataframe, user, dictionary_count):
    times = dataframe['timestamp']

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

    # importing the module
    # print(event_log)

    dict = {'ds': time, user: counts}

    for index, i in enumerate(dict):
        if i not in dictionary_count:
            dictionary_count[i] = dict[i]
    abc = pd.DataFrame(dictionary_count)
    abc.to_csv('Mycsv.csv')
    return dictionary_count


def convert_data_frame(dictionary):
    dataframe = pd.DataFrame(dictionary)
    return dataframe


def col_mean(df):
    columns = []
    for index, col in enumerate(df.columns):
        if index >= 1:
            columns.append(col)

    drop_columns = []
    for index, col1 in enumerate(columns):
        if index >= 1:
            drop_columns.append(col1)

    df['y'] = df[columns].mean(axis=1)
    df.drop(drop_columns, axis=1, inplace=True)
    return df


def model_prophet(df, dataframe1):
    model = Prophet(interval_width=0.99)

    # Fit the model on the training dataset
    model.fit(df)

    forecast = model.predict(df)

    # Visualize the forecast
    model.plot(forecast)
    df.tail(20)

    performance = pd.concat([df, forecast[['yhat', 'yhat_lower', 'yhat_upper']]], axis=1)
    dataframe1 = pd.concat([dataframe1, forecast[['yhat', 'yhat_lower', 'yhat_upper']]], axis=1)

    # performance = pd.merge(df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')

    columns = {'index': 0}
    for index, col in enumerate(dataframe1.columns):
        columns[col] = index + 1

    anom = []

    for row in dataframe1.itertuples():
        for i in range(2, columns['yhat']):
            if row[i] < row[columns['yhat_lower']] or row[i] > row[columns['yhat_upper']]:
                anom.append(i-1)
                break
        else:
            anom.append(0)

    dataframe1['anomaly'] = anom

    performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)

    # Check the number of anomalies
    performance['anomaly'].value_counts()

    anomalies = dataframe1[dataframe1['anomaly'] > 0].sort_values(by='ds')
    anom_performance = performance[performance['anomaly'] > 0].sort_values(by='ds')
    print(anom_performance)


if __name__ == '__main__':
    data = sys.argv
    data.pop(0)
    dictionary_user_1 = {}
    for user in data:
        file = user + '.csv'
        df1 = pd.read_csv(file)
        dictionary_user = df1.to_dict(orient='list')
        dictionary_user_1[user] = dictionary_user
    print('dic user 1',dictionary_user_1)
    dictionary_count_user = {}
    for user in dictionary_user_1:
        data_frame = read_file(user)
        hourly_frame_count_multi(data_frame, user, dictionary_count_user)
    print('dic count',dictionary_count_user)
    dataframe = convert_data_frame(dictionary_count_user)
    dataframe1 = dataframe.copy()
    df = col_mean(dataframe)

    model_prophet(df, dataframe1)
    with open('plotGraphMultiCount.pkl', 'wb') as f:
        pickle.dump([1,2,3,4,5], f, protocol=3)
