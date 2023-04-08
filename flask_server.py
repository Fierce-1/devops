import pickle
import subprocess
import json
import pandas as pd
import numpy as np
import csv
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/upload", methods=["POST"])
def upload():
    print("start uploading")
    file = request.files["file"]
    # save the file to the file system or process it in memory

    # Get the current directory path
    current_dir = os.getcwd()

    extension = os.path.splitext(file.filename)[1]
    if extension == '.csv':
        # Create the file path where you want to save the file
        file_path = os.path.join(current_dir, 'file'+extension)

        # Save the file
        file.save(file_path)

        unique_values = set()  # Create an empty set to store unique values
        header = pd.read_csv('file' + extension, nrows=0).columns.tolist()
        name_index = header.index('Account Name')
        with open('file' + extension, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                unique_values.add(row[name_index])  # Add the value in the first column to the set

        unique_list = list(unique_values)  # Convert the set to a list

        print(unique_list)  # Print the list of unique values
        response_body = json.dumps(unique_list)

        return response_body
    else:
        return jsonify({'error': 'Invalid File Extension'}), 500

@app.route("/graph", methods=["POST"])
def graph():
    print("In Graph")
    with open('plotGraphTime.pkl', 'rb') as f:
        plotGraph = pickle.load(f, encoding='latin1')
    response = json.dumps(plotGraph)
    print(response)
    return response


@app.route("/feature", methods=["POST"])
def feature():
    print("start working")
    start_date = 'False'
    end_date = 'False'
    single_user = request.form.get("singleUser") == "true"
    multi_user = request.form.get("multiUser") == "true"
    whole_date = False
    if request.form.get("wholeDate"):
        whole_date = True
    else:
        start_date = request.form.get("startDate")
        end_date = request.form.get("endDate")
    users = request.form.get("Users")
    time = request.form.get("time") == "true"
    count = request.form.get("count") == "true"
    pattern = request.form.get("pattern") == "true"
    users_list = users.split(',')

    # Call csv function so that we get dataframe of all users separately in dictionary
    users_dataframe_generator(users_list, start_date, end_date, whole_date)

    # Now will call the function in which we handle all other processes

    responseGraph = app.response_class(
        response=json.dumps(Complete_Remain_Process(users_list, single_user, multi_user, count, time, pattern)),
        status=200,
        mimetype='application/json'
    )
    return responseGraph


def users_dataframe_generator(users, time1, time2, wholeDate):
    global dic_user
    usersList = users
    startTime = time1
    endTime = time2
    df = pd.read_csv("file.csv")
    df['time'] = pd.to_datetime(df['datetime'])

    if not wholeDate:
        df = df.loc[(df['time'] >= startTime) & (df['time'] <= endTime)]
        print("here")

    df['timestamp'] = df["time"].values.astype(np.int64) / 1000000000
    cat = df['TaskCategory'].unique()
    user_dict = {}

    my_list = df.columns.values.tolist()

    acc_no = my_list.index('Account Name')

    task_cat = my_list.index('TaskCategory')

    tm = my_list.index('time')

    raw = my_list.index('raw')
    tm_stamp = my_list.index('timestamp')

    for user in usersList:
        diction = {'timestamp': [], 'message': []}
        for val_set in cat:
            diction[val_set] = []
        for idx, row in df.iterrows():
            if user in row[acc_no]:
                # diction['time'].append(row[tm])
                diction['message'].append(row[raw])
                diction['timestamp'].append(row[tm_stamp])
                add = row[task_cat]

                for val in diction:
                    if val == add:
                        diction[val].append(1)
                    elif val != 'time' and val != 'message' and val != 'timestamp':
                        diction[val].append(0)

        df1 = pd.DataFrame(diction)
        df1.to_csv(user + '.csv')


def Complete_Remain_Process(list_of_user, singleUser, multiUser, count, time, pattern):
    list_to_be_return = {}
    filename = ''
    pkl = ''
    dir = os.getcwd()
    for loop in range(0, 3):
        if singleUser:
            if count:
                filename = 'preprocessing_single_count_anomaly.py'
                pkl = 'plotGraphCount.pkl'
                count = False
            if time:
                filename = 'preprocessing_single_time_anomaly.py'
                pkl = 'plotGraphTime.pkl'
                time = False
            if pattern:
                filename = 'preprocessing_single_pattern.py'
                pkl = 'plotGraphPattern.pkl'
                pattern = False

            file_path = os.path.join(dir, filename)
            pklfile = os.path.join(dir, pkl)
            if filename:
                for user in list_of_user:
                    one_user = {}
                    subprocess.run(['python', file_path, user])
                    with open(pklfile, 'rb') as f:
                        plotGraph = pickle.load(f, encoding='latin1')

                    one_user['name'] = plotGraph[0]
                    one_user['count'] = plotGraph[1]
                    one_user['taskCategoryOverall'] = plotGraph[2]
                    one_user['eventperday'] = plotGraph[3]
                    list_to_be_return[user] = one_user
            filename = ""
            print(list_to_be_return)

        if multiUser:
            if count:
                filename = 'preprocessing_multi_user_count.py'
                count = False
                pkl = 'plotGraphMultiCount.pkl'
            if time:
                filename = 'preprocessing_multi_user_time.py'
                time = False
                pkl = 'plotGraphTime.pkl'
            if pattern:
                filename = 'preprocessing_single_pattern.py'
                pattern = False
                pkl = 'plotGraphPattern.pkl'

            file_path = os.path.join(dir, filename)
            subprocess.run(['python', file_path, *list_of_user])
        print(list_of_user)

    return list_to_be_return


if __name__ == '__main__':
    app.run()

