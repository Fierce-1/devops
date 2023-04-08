import ast
import pickle
import sys
from sent2vec.vectorizer import Vectorizer
import pandas as pd
from sklearn.cluster import DBSCAN
from numpy import where
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from kneed import KneeLocator


def read_file():
    data = sys.argv[1]
    df = pd.read_csv(data + '.csv')
    return df


def file_read_pattern():
    data = sys.argv[1]
    df = pd.read_csv(data+'.csv')
    data = list(df['message'])
    return data


def pattern_parder():
    data = file_read_pattern()
    patt4 = []

    account_name = 'Account Name'
    things = ['EventCode', 'TaskCategory', 'Message', 'Impersonation Level', 'Token Elevation Type', 'Source Address',
              'Source Port', 'Logon Type', 'OpCode', 'Type', 'Keywords', 'Subject ID', 'Object Type', 'Access Mask',
              'Accesses', 'Share Name', 'Share Path', 'New Process Name', 'Process Name', 'Creator Process Name',
              'Process Command Line', 'Object Server', 'Access Reasons', 'Privileges Used for Access Check',
              'Resource Attributes', 'Enabled Privileges', 'Privileges', 'Relative Target Name', 'SYNCHRONIZE',
              'ReadAttributes', 'Logon Process Name', 'Group Name', 'Group Domain', 'Name',
              'Original Security Descriptor', 'New Security Descriptor']

    diction = {}
    count = []
    a = 0
    for patt in data:

        patt2 = patt.replace('\n', '|')
        patt2 = patt2.replace(':|\t', '|')
        patt2 = patt2.replace(':', '|')
        patt2 = patt2.replace('=', '|')
        patt2 = patt2.replace('\t', '')
        patt3 = patt2.split('|')
        index_account = patt3.index(account_name) + 1
        name = patt3[index_account]

        msg = ''
        for j in things:
            if j in patt3:
                index = patt3.index(j) + 1
                msg = msg + ' ' + patt3[index]

        patt4.append(msg)
        count.append(a)
        a += 1
    diction['msg'] = patt4
    diction['Count'] = count
    # print(len(diction['msg']))

    df2 = pd.DataFrame(diction)
    return df2


def sent_to_vec():
    data2 = pattern_parder()

    vectorizer = Vectorizer()
    sentence = list(data2['msg'])

    vectorizer.run(sentence)
    v_bert = vectorizer.vectors

    return v_bert


def DBSCAN_MODEL():
    arr = np.array(sent_to_vec())
    # Compute pairwise distances between all data points
    distances = pdist(arr)

    # Perform hierarchical clustering with a linkage method of your choice
    linkage_matrix = linkage(distances, method='ward')

    # Compute the distances between the cluster centroids at each level of the dendrogram
    cluster_distances = linkage_matrix[:, 2]

    # Find the knee point in the distances plot
    kneedle = KneeLocator(range(len(cluster_distances)), cluster_distances, S=1.0, curve="convex",
                          direction="increasing")
    eps = kneedle.elbow_y

    print("The optimal eps value is:", eps)

    model = DBSCAN(algorithm='auto', eps=eps, leaf_size=30, metric='euclidean',
                   metric_params=None, min_samples=2, n_jobs=None, p=None)

    # We'll fit the model with x dataset and get the prediction data with the fit_predict() method.
    # arr = np.array(display_arr)

    pred = model.fit_predict(arr)

    # Next, we'll extract the negative outputs as the outliers.

    anom_index = where(pred == -1)

    values = arr[anom_index]

    return anom_index


'''
*************************************************************
BELOW THIS IS THE CODE FOR EVENTS PER DAY AND CATEGORY COUNT
*************************************************************
'''


def get_columns():
    df = read_file()
    columns = {}
    for index, col in enumerate(df.columns):
        columns[col] = index
    return columns


def result_for_total_category(colName):
    df1 = read_file()
    category = []
    counts = []
    for cat in colName:
        if cat != 'Unnamed: 0' and cat != 'timestamp' and cat != 'message':
            Total = df1[cat].sum()
            if Total != 0:
                category.append(cat)
                counts.append(int(Total))
    taskCategoryOverall = {'category': category, 'total': counts}
    return taskCategoryOverall

def result_for_events_per_day():
    dataframe_for_eventsperDay = read_file()

    # convert timestamps to dates and get count of unique dates
    dates = pd.to_datetime(dataframe_for_eventsperDay['timestamp'], unit='s').dt.date
    counts = dates.value_counts()

    # create dictionary with date values and their counts
    eventperday = {'date': pd.DatetimeIndex(counts.index).strftime('%Y-%m-%d').tolist(), 'total': counts.tolist()}

    return eventperday


colName = get_columns()
taskCategoryOverall = result_for_total_category(colName)
eventperday = result_for_events_per_day()


anomaly = DBSCAN_MODEL()
print(anomaly)
arr = np.array(anomaly)
arr = arr.flatten()
arr = arr.tolist()
count = len(arr)
with open('plotGraphPattern.pkl', 'wb') as f:
    pickle.dump([arr, count, taskCategoryOverall, eventperday], f, protocol=3)
